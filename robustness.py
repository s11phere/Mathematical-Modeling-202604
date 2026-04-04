# 该文档中可调用的函数
"""
max_link(graph_dic)  输入一个字典图输出最大联通
pop_p(graph_dic,hitler_list) 输入字典图与踢点集合,输出踢点后的新图
strenth_line(graph_dic,N,d=150) 输入字典图与蒙特卡洛次数与分辨率,默认150,计算效能曲线,输出y值列表
plot_strenth_curve(city_name, l, output_dir="performance_data",compare = False,adjusted = False) 传入城市名字,数据列表,存储路径,是否为方格参照,是否标准化，绘制性能曲线
robustness(l) 鲁棒性函数,传入性能曲线,计算鲁棒性
creat_square_grid(n) 用于对比,生成边长为n的方形网格,输出为字典
"""


import os
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import math
from q1 import get_processed_graph






def creat_square_grid(n):
    """
    生成一个 n x n 的正方形网格图（无向图）。
    节点编号：id = r * n + c，其中 r, c 范围 0 到 n-1。
    每个节点与其上下左右邻居相连（如果存在）。
    
    参数:
        n (int): 网格边长（节点数 = n * n）
    
    返回:
        graph_dic (dict): 邻接表，如 {node_id: [neighbor_id, ...]}
    """
    graph_dic = {}
    total_nodes = n * n
    
    # 为每个节点生成邻居列表
    for r in range(n):
        for c in range(n):
            node_id = r * n + c
            neighbors = []
            # 上
            if r > 0:
                neighbors.append((r - 1) * n + c)
            # 下
            if r < n - 1:
                neighbors.append((r + 1) * n + c)
            # 左
            if c > 0:
                neighbors.append(r * n + (c - 1))
            # 右
            if c < n - 1:
                neighbors.append(r * n + (c + 1))
            graph_dic[node_id] = neighbors
    
    return graph_dic



def read_road_network(csv_file_path):
    """
    读取道路网络CSV文件，返回邻接表字典。
    
    参数:
        csv_file_path (str): CSV文件路径
    
    返回:
        graph_dict (dict): 邻接表，例如 {node: [neighbor1, neighbor2, ...]}
    """
    df = pd.read_csv(csv_file_path)
    
    # 检查必要的列是否存在
    required_columns = ['START_NODE', 'END_NODE']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")
    
    # 使用 set 避免重复边
    graph = defaultdict(set)
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        graph[u].add(v)
        # 如需无向图（计算连通分量时常用），取消下面一行的注释
        # graph[v].add(u)
    
    # 将 set 转换为 list 以便后续使用
    graph_dict = {k: list(v) for k, v in graph.items()}
    return graph_dict


def pop_p(graph_dic,hitler_list): # 输入待踢点地图与踢点集合
    hitler_list = set(hitler_list) # 接受列表,集合均可
    graph_new = {}
    for nod in graph_dic.keys():
        if nod in hitler_list:
            continue
        else:
            l = []
            for i in graph_dic[nod]:
                if i in hitler_list:
                    continue
                else:
                    l.append(i)
            graph_new[nod] = l
    return(graph_new)



def strenth_line(graph_dic,N,d=150): # 效能曲线绘制函数，N为蒙特卡洛次数
    size = len(graph_dic)

    l = [0]*(d+1) # 存储数据点的函数
    size = len(graph_dic) # 节点个数
    max_link0 = max_link(graph_dic)



    x = [int(size*i/d) for i in range(d + 1)] # 每个绘图点去掉了的节点个数

    for t in range(N):
        temp_graph = graph_dic
        for i in range(d+1):
            if i == 0:
                l[0] += 1/N
                continue
            elif i == d:
                l[i] += 0
                continue
            else:
                n_pop = x[i]-x[i-1]
                hitler_set = set(random.sample(list(temp_graph.keys()),n_pop))
                temp_graph = pop_p(temp_graph,hitler_set)
                l[i] += max_link(temp_graph)/(max_link0*N)
        print("第",t,"次蒙特卡洛循环")
    return(l)


def adjust(l):
    d = len(l)-1
    for i in range(d+1):
        if i == d:
            continue
        else:
            l[i] = l[i]/(1-i/d)
    return(l)

    
def plot_strenth_curve(city_name, l, output_dir="performance_data",compare = False,adjusted = False):
    """
    根据计算得到的列表 l 绘制鲁棒性曲线，并保存 CSV 数据和图片。
    """
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    d = len(l)-1
    x = [i / d for i in range(d + 1)]   # 横坐标：移除比例
    y = l
    

    # 保存 CSV
    df = pd.DataFrame({"Fraction_Removed": x, "Normalized_Size": y})
    csv_path = output_path / f"{city_name}_strenth_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"数据已保存至: {csv_path}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=1.5, markersize=3)
    if compare and not adjusted:
        plt.title(f"Network Strenth Curve - {city_name}_compare")
    elif not compare and not adjusted:
        plt.title(f"Network Strenth Curve - {city_name}")
    elif compare and adjusted:
        plt.title(f"Network Strenth Curve - {city_name}_compare_adjusted")
    elif not compare and adjusted:
        plt.title(f"Network Strenth Curve - {city_name}_adjusted")
    plt.xlabel("Fraction of Removed Nodes")
    plt.ylabel("Normalized Largest Component Size")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    if compare and not adjusted:
        img_path = output_path / f"{city_name}_compare_strenth_curve.png"
    elif compare and adjusted:
        img_path = output_path / f"{city_name}_compare_strenth_curve_adjusted.png"
    elif not compare and not adjusted:
        img_path = output_path / f"{city_name}_strenth_curve.png"
    elif not compare and adjusted:
        img_path = output_path / f"{city_name}_strenth_curve_adjusted.png"
    plt.savefig(img_path, dpi=300)
    print(f"图片已保存至: {img_path}")
    plt.show()


# 对于一张给定图判断最大联通分量
def max_link(graph_dic): # 输入为一个存储了节点的字典
    linked_group = [] # 用于存储相互连接的点集
    visited = set()
    max_size = 0
    
    for nod in graph_dic.keys():
        if nod in visited:
            continue
        else: # 深搜扩展
            tep_size = 1
            stack = []
            visited.add(nod)
            for son in graph_dic[nod]:
                if not son in visited:
                    stack.append(son)
                    visited.add(son)
            while stack:
                cur = stack.pop()
                tep_size += 1
                for grandson in graph_dic[cur]:
                    if grandson in visited:
                        continue
                    else:
                        stack.append(grandson)
                        visited.add(grandson)
        if tep_size > max_size:
            max_size = tep_size
    return(max_size)



def robustness(l):
    return(sum(l)/(len(l)-1))



def main():
    city_files = [
        "cases/Chengdu_Edgelist.csv",
        "cases/Dalian_Edgelist.csv",
        "cases/Dongguan_Edgelist.csv",
        "cases/Harbin_Edgelist.csv",
        "cases/Qingdao_Edgelist.csv",
        "cases/Quanzhou_Edgelist.csv",
        "cases/Shenyang_Edgelist.csv",
        "cases/Zhengzhou_Edgelist.csv"
    ]

    robustness_data = {}

    for city_file in city_files:
        csv_path = Path(__file__).parent / city_file
        city_name = csv_path.stem.replace('_Edgelist', '')

        print(f"\n========== 处理 {city_name} ==========")
        # 读取原始数据
        df = pd.read_csv(csv_path)
        # 调用清洗函数
        G,_ = get_processed_graph(df, city_name=city_name, plot=False, include_length=False)
        # 转换为无向图（兼容带权邻接表）
        print(type(G))
        for key in G.keys():
            if G[key] != None:
                print(key)


        print(f"成功读取并清洗 {city_name} 道路记录")
        print(f"清洗后节点数: {len(G)}")
        print(f"最大连通分量: {max_link(G)}")

        l = strenth_line(G, N=20)
        plot_strenth_curve(city_name, l)

        print(f"{city_name} 鲁棒性: {robustness(l)}")

        l_ad = adjust(l)
        plot_strenth_curve(city_name, l_ad, compare=False, adjusted=True)
        robustness_data[city_name] = robustness(l)

    print(robustness_data)
    print("\n所有城市处理完成！")


def related():
    city_files = [
        "cases/Chengdu_Edgelist.csv",
        "cases/Dalian_Edgelist.csv",
        "cases/Dongguan_Edgelist.csv",
        "cases/Harbin_Edgelist.csv",
        "cases/Qingdao_Edgelist.csv",
        "cases/Quanzhou_Edgelist.csv",
        "cases/Shenyang_Edgelist.csv",
        "cases/Zhengzhou_Edgelist.csv"
    ]

    compare_robustness_data = {}

    for city_file in city_files:
        csv_path = Path(__file__).parent / city_file
        city_name = csv_path.stem.replace('_Edgelist', '')

        print(f"\n========== 处理 {city_name} (网格对比) ==========")
        df = pd.read_csv(csv_path)
        G,_= get_processed_graph(df, city_name=city_name, plot=False, include_length=False)

        print(f"成功读取并清洗 {city_name} 道路记录")
        print(f"清洗后节点数: {len(G)}")
        print(f"最大连通分量: {max_link(G)}")

        n = int(math.sqrt(len(G)))
        grid = creat_square_grid(n)

        grid_line = strenth_line(grid, 20)
        grid_robust = robustness(grid_line)

        compare_robustness_data[city_name] = grid_robust

        plot_strenth_curve(city_name, grid_line, output_dir="performance_data", compare=True)

        grid_line_ad = adjust(grid_line)

        plot_strenth_curve(city_name, grid_line_ad, output_dir="performance_data", compare=True, adjusted=True)

    print(compare_robustness_data)



def stablity(G):
    return


if __name__ == "__main__":

    main()
    related()