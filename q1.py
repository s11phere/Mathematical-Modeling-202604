import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy import stats
import networkx as nx
import os

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

# 存储每个城市的统计结果
results = {}

def build_graph(df):
    """建立有向图"""
    neighbors = {}
    for idx, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        neighbors.setdefault(u, []).append(v)
    for node in neighbors:
        neighbors[node] = list(set(neighbors[node]))
    return neighbors

def degree_dist(neighbors):
    """计算度序列及度分布频数"""
    deg = [len(neigh) for neigh in neighbors.values()]
    deg_counter = Counter(deg)
    return deg, deg_counter

def cluster(neighbors):
    """计算平均聚类系数"""
    cluster_coef = {}
    for cur, neighbor in neighbors.items():
        n = len(neighbor)
        if n < 2:
            cluster_coef[cur] = 0
            continue
        max_possible = n * (n - 1) / 2
        actual = 0
        for i in range(n):
            u = neighbor[i]
            for j in range(i+1, n):
                v = neighbor[j]
                if v in neighbors.get(u, []):
                    actual += 1
        cluster_coef[cur] = actual / max_possible
    avg_c = np.mean(list(cluster_coef.values())) if cluster_coef else 0
    return avg_c, cluster_coef

# def diameter(neighbors):
#     """计算最大连通分量的直径"""
#     G = nx.Graph()
#     for node, nbrs in neighbors.items():
#         for nbr in nbrs:
#             G.add_edge(node, nbr)
#     # 获取最大连通分量
#     if not nx.is_connected(G):
#         components = list(nx.connected_components(G))
#         largest_comp = max(components, key=len)
#         G_largest = G.subgraph(largest_comp).copy()
#         diameter = nx.diameter(G_largest)
#     else:
#         diameter = nx.diameter(G)
#     return diameter

for filepath in city_files:
    city_name = os.path.basename(filepath).replace("_Edgelist.csv", "")
    print(f"\nCurrent City: {city_name}")
    df = pd.read_csv(filepath)
    
    # 构建无向邻居字典
    neighbors = build_graph(df)
    node_cnt = len(neighbors)
    print(f"  节点数: {node_cnt}")
    
    # 度分布
    deg_seq, deg_cnt = degree_dist(neighbors)
    max_deg = max(deg_seq)
    min_deg = min(deg_seq)
    print(f"  最小度: {min_deg}, 最大度: {max_deg}")
    print("  度分布:")
    for d in sorted(deg_cnt):
        print(f"    deg {d}: {deg_cnt[d]}")
    
    # 聚类系数
    avg_c, cluster_coef_dict = cluster(neighbors)
    print(f"  平均聚类系数: {avg_c:.4f}")
    
    # 直径
    # diam = diameter(neighbors)
    # print(f"  直径（最大连通分量）: {diam}")
    
    # 保存结果
    results[city_name] = {
        "nodes": node_cnt,
        "min_deg": min_deg,
        "max_deg": max_deg,
        "deg_dist": deg_cnt,
        "avg_clustering": avg_c,
        # "diameter": diam,
        "cluster_coef_dict": cluster_coef_dict
    }
    
    # 绘制度分布直方图
    plt.figure()
    deg_vals = list(deg_cnt.keys())
    freq = list(deg_cnt.values())
    plt.bar(deg_vals, freq, width=0.8)
    plt.xlabel("Degree")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Degree Distribution - {city_name}")
    plt.savefig(f"figures/{city_name}_deg_dist.png", dpi=150)
    plt.close()
    
    # 绘制聚类系数分布直方图
    cluster_vals = list(cluster_coef_dict.values())
    plt.figure()
    plt.hist(cluster_vals, bins=np.arange(0, 1.00, 0.02))
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Count")
    plt.title(f"Clustering Coefficient Distribution - {city_name}")
    plt.savefig(f"figures/{city_name}_clust_dist.png", dpi=150)
    plt.close()

# 汇总输出
print("\n===== result ======")
summary_df = pd.DataFrame([
    {
        "City": city,
        "Nodes": info["nodes"],
        "MinDeg": info["min_deg"],
        "MaxDeg": info["max_deg"],
        "AvgClust": info["avg_clustering"],
        # "Diameter": info["diameter"],
        # "PowerLawAlpha": info["powerlaw_alpha"] if info["powerlaw_alpha"] else np.nan
    }
    for city, info in results.items()
])
print(summary_df.to_string(index=False))
summary_df.to_csv("network_summary.csv", index=False)