import os
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import math
from q1 import get_processed_graph
import statistics
import heapq
import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import networkx as nx
from functools import partial
import copy
from datetime import datetime




"""
get_data(city_name:str) 接受城市名,输出nx地图
get_path(G,start:int,end:int) 接受图,起始点,输出一个路径集合
remove_node_from_nx(G: nx.Graph, l) 删除G中的一组节点,返回G
max_link_nx(G: nx.Graph) 获取G的max-link
k_core_choose(G: nx.Graph,n:int) 获取G的k-核中要删除的n个节点,以列表返回
attack(G: nx.Graph, l:list) 输入G与攻击顺序列表,返回G的攻击结果
plot_list(city_name, l,n:int,N:int, output_dir:str) 绘图函数,传入城市名字,下降过程列表(需归一化),批尺寸,图尺寸,储存路径(一个词)
simulation(G, city_name: str, func, n: int, lower_bound=0.01,strategy_name=None) 核心模拟函数,传入G,城市名字,攻击函数,批尺寸,下界,攻击策略名
list_attack_resout(G,attack_list:list,d:int) 用于进化,传入城市地图与名字与攻击列表和攻击的分辨率,返回攻击下的鲁棒性
expected_robustness(G,records: list, lower_bound=0.01) 用于进化,传入删点记录,求截止过的鲁棒性
"""


def get_data(city_name:str): 
    csv_path = Path(__file__).parent / "cases" / f"{city_name}_filtered_edgelist.csv"
    df = pd.read_csv(csv_path)
    required_cols = ['START_NODE', 'END_NODE', 'LENGTH']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV 缺少必需的列: START_NODE, END_NODE, LENGTH")

    G = nx.Graph()

    # 记录每个节点的坐标（取第一次出现）
    node_coords = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        # 处理坐标（可选）
        if 'XCoord' in df.columns and 'YCoord' in df.columns:
            if u not in node_coords:
                node_coords[u] = (row['XCoord'], row['YCoord'])
            if v not in node_coords:
                node_coords[v] = (row['XCoord'], row['YCoord'])  # 注意：同一行的两个节点坐标可能不同，这里简化处理
        # 添加无向边（标准化方向，避免重复）
        u, v = sorted([u, v])  # 确保 (min, max)
        length = row['LENGTH']
        if G.has_edge(u, v):
            # 如果边已存在（例如双向记录），累加长度
            G[u][v]['length'] += length
        else:
            G.add_edge(u, v, length=length)

    # 添加节点坐标属性
    for node, (x, y) in node_coords.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y

    print(city_name,"获取成功")
    return(G)    





def get_path(G: nx.Graph, start: int, end: int) -> set:

    try:
        path = nx.shortest_path(G, source=start, target=end)
        return set(path)
    except nx.NetworkXNoPath:
        return set()
    


def remove_node_from_nx(G: nx.Graph, l):
    """从 networkx 图中删除节点（及其所有边）"""
    for node in l:
        G.remove_node(node)

def max_link_nx(G: nx.Graph) -> int:
    """返回 networkx 图的最大连通分量的节点数"""
    if G.number_of_nodes() == 0:
        return 0
    return max(len(c) for c in nx.connected_components(G))



def k_core_choose(G: nx.Graph,n:int,k=2):
    """
    从图 G 的 k-核中，按原始图度数降序选择 n 个节点，度数相同时随机排序。

    参数
    ----------
    G : nx.Graph
        输入图
    n : int
        需要选择的节点数量
    k : int, default=2
        核的阶数

    返回
    -------
    list
        选中的节点列表（长度 ≤ n）
    """
    # 获取 k-核的节点集合
    core_G = nx.k_core(G, k=k)
    core_nodes = set(core_G.nodes)
    if not core_nodes:
        all_nodes = list(G.nodes())
        if len(all_nodes) <= n:
            return all_nodes
        return(list(random.sample(all_nodes, n)))

    # 生成 (度数, 随机数, 节点) 三元组，便于堆排序时同度数随机
    items = [(core_G.degree(node), random.random(), node) for node in core_nodes]
    
    # 使用堆取前 n 个最大（按度数降序，同度数按随机数）
    top_items = heapq.nlargest(n, items)
    
    # 提取节点，保持顺序（已按度数降序，同度数随机）
    selected = [node for (_, _, node) in top_items]
    
    return(selected)



def attack(G: nx.Graph, l: list):
    """从图中删除列表 l 中的所有节点"""
    G.remove_nodes_from(l)
    return G




def plot_list(city_name, l, n, N, output_dir):
    """
    根据计算得到的列表 l 绘制鲁棒性曲线，并保存 CSV 数据和图片。
    output_dir: 子目录名（如 "2_core", "3_core", "betweenness"）
    """
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    d = len(l) - 1
    x = [n * i / N for i in range(d + 12)]   # 横坐标：移除比例
    y = l + [0] * 11

    # 保存 CSV
    csv_path = output_path / f"{city_name}_{output_dir}_attack.csv"
    df = pd.DataFrame({"Fraction_Removed": x, "Normalized_Size": y})
    df.to_csv(csv_path, index=False)
    print(f"数据已保存至: {csv_path}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=1.5, markersize=3)
    plt.title(f"Network Strength Curve - {city_name} ({output_dir})")
    plt.xlabel("Fraction of Removed Nodes")
    plt.ylabel("Normalized Largest Component Size")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(x))
    plt.ylim(0, 1)
    plt.tight_layout()

    img_path = output_path / f"{city_name}_{output_dir}_attack.png"
    plt.savefig(img_path, dpi=300)
    print(f"图片已保存至: {img_path}")
    plt.show()


def simulation(G, city_name: str, func, n: int, lower_bound=0.01, strategy_name=None):
    """
    攻击模拟函数
    strategy_name: 可选，用于指定输出子目录名。若不提供，则根据 func 自动推断。
    """
    G = G.copy()
    N = G.number_of_nodes()
    if N == 0:
        return

    records = [1.0] # 结果记录
    removed_total = 0



    while True:
        to_remove = func(G, n)
        if not to_remove:
            break
        G.remove_nodes_from(to_remove)
        removed_total += len(to_remove)
        if G.number_of_nodes() == 0:
            largest = 0
        else:
            largest = max(len(c) for c in nx.connected_components(G))
        norm = largest / N
        records.append(norm)


        if norm <= lower_bound or G.number_of_nodes() == 0:
            break

    # 确定输出子目录
    if strategy_name is not None:
        output_dir = strategy_name
    elif func == k_core_choose:
        # 默认 k=2，但如果调用时使用了 functools.partial 或 lambda，则无法直接获取 k
        # 简单起见：要求调用 simulation 时传入 strategy_name
        output_dir = "2_core"
    else:
        output_dir = func.__name__

    plot_list(city_name, records, n, N, output_dir)

    robustness = sum(records)*n/N
    print(city_name,"robustness",robustness,"完成")

    return(robustness)


    




def list_attack_resout(G,attack_list:list,d=40):
    G = G.copy()
    N = G.number_of_nodes()
    if N == 0:
        return(0)

    records = [1.0] # 结果记录
    
    divide_point = [(i * len(attack_list)) // d for i in range(d + 1)]

    for i in range(d):
        to_remove = attack_list[divide_point[i]: divide_point[i + 1]]
        if not to_remove:
            break
        G.remove_nodes_from(to_remove)
        
        if G.number_of_nodes() == 0:
            largest = 0
        else:
            largest = max(len(c) for c in nx.connected_components(G))
        norm = largest / N
        records.append(norm)


    return(sum(records)/(4*d),records)



def expected_robustness(G,records: list, lower_bound=0.01):
    N = G.number_of_nodes()
    if min(records) >= lower_bound:
        return(sum(records) / N)
    else:
        return(sum(records[i] for i in range(len(records)) if records[i] >= lower_bound)/N)




def initialize_population(G, pop_size=20):
    """初始化种群"""
    L = int(G.number_of_nodes() / 4)
    population = []
    for _ in range(pop_size):
        individual = list(random.sample(list(G.nodes()), L))
        population.append(individual)
    return(population)



def cross_over(cross_c_rate,parent1, parent2):
    """交叉操作"""
    delta = random.uniform(0, 1)
    if delta < cross_c_rate:
        L = len(parent1)
        child = []
        child += parent1[:3*L // 5]
        child_set = set(child)
        i = 0
        while len(child) < L:
            if parent2[i] not in child_set:
                child.append(parent2[i])
                child_set.add(parent2[i])
            i += 1
        return(child)
    else:
        return(parent1.copy())

    


def mutate(individual, G, mutation_rate, replace_k_range=(20,80), shuffle_l_range=(100,200)):
    """变异操作"""
    L = len(individual)
    individual_set = set(individual)

    # 替换操作
    delta = random.uniform(0, 1)
    if delta < mutation_rate:
        replace_k = random.randint(*replace_k_range)
        for _ in range(replace_k):
            node = random.choice(list(G.nodes()))
            while node in individual_set:  # 避免重复
                node = random.choice(list(G.nodes()))
            index = random.randint(0, L - 1)
            individual[index] = node
            individual_set.add(node)

    # 随机交换操作
    delta = random.uniform(0, 1)
    if delta < mutation_rate:
        shuffle_l = random.randint(*shuffle_l_range)
        start = random.randint(0, L - shuffle_l)
        end = start + shuffle_l
        individual_shuffle = individual[start:end]
        random.shuffle(individual_shuffle)
        individual[start:end] = individual_shuffle
    
    return(individual)



def select(population, fitnesses, tournament_size=3):
    """锦标赛选择，使用预先计算的适应度列表"""
    candidates_idx = random.sample(range(len(population)), tournament_size)
    best_idx = min(candidates_idx, key=lambda i: fitnesses[i])
    return population[best_idx]




def evaluate_population(population, G, d):
    fitnesses = []
    for ind in population:
        area, _ = list_attack_resout(G, ind, d)
        fitnesses.append(area)
    return fitnesses




def evolve(G,pop_size=20,generations=100,cross_c_rate=0.5,mutation_rate=0.1,replace_k_range=(20,80),
           shuffle_l_range=(100,200),d=40,elite_size=5,tournament_size=3,verbose = False):
    N = G.number_of_nodes()
    L = int(N / 4)

    population = initialize_population(G, pop_size)
    best_individual = None
    best_fitness = float("inf")
    fitness_history = []

    for gen in range(generations):

        fitnesses = evaluate_population(population, G, d)

        gen_best_fitness = min(fitnesses)
        if gen_best_fitness < best_fitness:
            best_individual = population[fitnesses.index(gen_best_fitness)].copy()
            best_fitness = gen_best_fitness

        fitness_history.append(gen_best_fitness)

        if verbose:
            print("Best individual:", best_individual)
            print("Best fitness:", best_fitness)
            print("Fitness history:", fitness_history)

        # 精英保留
        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
        new_population = [population[i] for i in elite_indices]

        while len(new_population) < pop_size:
            parent1 = select(population, fitnesses, tournament_size)
            parent2 = select(population, fitnesses, tournament_size)
            child = cross_over(cross_c_rate, parent1, parent2)
            child = mutate(child, G, mutation_rate, replace_k_range, shuffle_l_range)
            new_population.append(child)

        population = new_population

        print("第",gen+1,"代完成")
        print("Best fitness:", best_fitness)


    return(best_individual, best_fitness, fitness_history)







def plot_fitness_curve(fitness_history, city_name, output_dir="evolution_history"):
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{city_name}_{output_dir}_evolution_history_{timestamp}.png"
    save_path = output_path / filename
    
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, linewidth=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Robustness Area)")
    plt.title(f"GA Optimization - {city_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"进化曲线已保存至: {save_path}")
    plt.show()





def plot_best_attack(G, best_seq, d, city_name, output_dir="ga_results"):
    area, records = list_attack_resout(G, best_seq, d)
    N = G.number_of_nodes()
    m = len(best_seq)
    x = [i * d / N for i in range(len(records))]   # 实际删除比例

    script_dir = Path(__file__).parent
    out_path = script_dir / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = out_path / f"{city_name}_optimized_attack_{timestamp}.png"
    csv_path = out_path / f"{city_name}_optimized_attack_{timestamp}.csv"

    # 保存 CSV
    pd.DataFrame({"Fraction_Removed": x, "Normalized_Size": records}).to_csv(csv_path, index=False)

    # 绘图
    plt.figure(figsize=(10,6))
    plt.plot(x, records, marker='o', linestyle='-')
    plt.title(f"Optimized Attack - {city_name} (Area={area:.4f})")
    plt.xlabel("Fraction of Removed Nodes")
    plt.ylabel("Normalized Largest Component")
    plt.grid(True)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.show()
    print(f"图片保存至: {img_path}")
    return area











if __name__ == "__main__":
    city_names = ["Chengdu","Dalian","Dongguan","Harbin","Qingdao","Quanzhou","Shenyang","Zhengzhou"]

    city_name = "Qingdao"

    G = get_data(city_name)   # 您的读取函数
    best_seq, best_fit, hist = evolve(G, pop_size=20, generations=1000, verbose=False)
    plot_fitness_curve(hist, city_name)
    plot_best_attack(G, best_seq, 40, city_name)   # 使用新实现的不覆盖版本
        
