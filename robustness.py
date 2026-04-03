# 该文档中可调用的函数
"""
max_link(graph_dic)  输入一个字典图输出最大联通
pop_p(graph_dic,p) 输入字典图与概率,输出踢点后的新图
strenth_line(graph_dic,N, city_name=None, output_dir="performance_data") 输入字典图与蒙特卡洛次数和城市名字,可视化效能曲线,输出y值列表
"""


import os
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import random
import matplotlib.pyplot as plt

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


def pop_p(graph_dic,p): # 输入待踢点地图与踢点概率
    n_tot = len(graph_dic)
    n_pop = int(n_tot*p)
    hitler_list = set(random.sample(list(graph_dic.keys()),n_pop)) # 待踢点集合
    
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



def strenth_line(graph_dic,N, city_name=None, output_dir="performance_data"): # 效能曲线绘制函数，N为蒙特卡洛次数
    size = len(graph_dic)
    l = []
    d = 150 # 超参数d决定分辨率

    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)   # 创建目录




    for i in range(0,d+1):
        temp = 0
        print("p=",i/d)
        if i == 0:
            l.append(max_link(graph_dic)/size)
            continue
        elif i == d :
            l.append(0)
            continue
        else:
            for _ in range(N):
                poped_graph = pop_p(graph_dic,i/d)
                temp += max_link(poped_graph)/size
            l.append(temp/N)

    x = [i / d for i in range(d + 1)]
    y = l
    # 保存 CSV
    df = pd.DataFrame({"Fraction_Removed": x, "Normalized_Size": y})
    csv_path = output_path / f"{city_name}_robustness_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"数据已保存至: {csv_path}")
    
    # 绘图并保存图片
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=1.5, markersize=3)
    plt.title(f"Network Robustness Curve - {city_name}")
    plt.xlabel("Fraction of Removed Nodes")
    plt.ylabel("Normalized Largest Component Size")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    img_path = output_path / f"{city_name}_robustness_curve.png"
    plt.savefig(img_path, dpi=300)
    print(f"图片已保存至: {img_path}")
    plt.show()
    
    return l


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




def main():
    csv_file = Path(__file__).parent / 'cases' / 'Shenyang_Edgelist.csv'
    # 提取城市名（例如 "Qingdao"）
    city_name = csv_file.stem.replace('_Edgelist', '')
    
    G = read_road_network(csv_file)
    print(f"成功读取 {city_name} 道路记录")
    print(f"节点数: {len(G)}")
    print(f"最大连通分量: {max_link(G)}")
    
    strenth_line(G, N=20, city_name=city_name)

if __name__ == "__main__":
    main()