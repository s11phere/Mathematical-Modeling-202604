import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter,deque
from scipy import stats
import os
from networkx.algorithms import approximation
import heapq

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

def build_graph(df:pd.DataFrame):
    """建立有向图"""
    neighbors = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        neighbors.setdefault(u, []).append(v)
    for node in neighbors:
        neighbors[node] = list(set(neighbors[node]))
    return neighbors

def degree_dist(neighbors:dict):
    """计算度序列及度分布频数"""
    deg = [len(neigh) for neigh in neighbors.values()]
    deg_counter = Counter(deg)
    return deg, deg_counter

def cluster(neighbors:dict):
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

def dijkstra(src, graph):
    """返回 (dist_list, farthest_node, max_dist)"""
    n = len(graph)
    dist = {}
    for node in graph:
        dist[node]=float('inf')
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    max_dist = max(dist.values())
    farthest = next(node for node, d in dist.items() if d == max_dist)
    return farthest, max_dist

def compute_diameter(df:pd.DataFrame):
    """计算直径，考虑最大连通分量"""
    G={}
    for _,row in df.iterrows():
        u=row['START_NODE']
        v=row['END_NODE']
        w=row['LENGTH']
        G.setdefault(u, []).append((v, w))
        G.setdefault(v, []).append((u, w))

    # 寻找最大连通分量
    visited = set()
    max_comp_nodes = set()
    max_size = 0
    for node in G:
        if node in visited:
            continue
        comp_nodes = set()
        q = deque([node])
        visited.add(node)
        comp_nodes.add(node)
        while q:
            cur = q.popleft()
            for nbr, _ in G[cur]:
                if nbr not in visited:
                    visited.add(nbr)
                    comp_nodes.add(nbr)
                    q.append(nbr)
        if len(comp_nodes) > max_size:
            max_size = len(comp_nodes)
            max_comp_nodes = comp_nodes

    # 构建最大连通分量的子图
    max_graph = {node: G[node] for node in max_comp_nodes}
    print(f"total_nodes: {len(G)}, max_node_cnt: {max_size}, connected: {len(G)==max_size}")

    # 计算直径
    s = next(iter(max_graph))
    nd1,_=dijkstra(s,max_graph)
    _,diameter=dijkstra(nd1,max_graph)

    return diameter

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
    diam = compute_diameter(df)
    print(f"  直径（最大连通分量）: {diam}")
    
    # 保存结果
    results[city_name] = {
        "nodes": node_cnt,
        "min_deg": min_deg,
        "max_deg": max_deg,
        "deg_dist": deg_cnt,
        "avg_clustering": avg_c,
        "diameter": diam,
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
        "Diameter": info["diameter"],
    }
    for city, info in results.items()
])
print(summary_df.to_string(index=False))
summary_df.to_csv("network_summary.csv", index=False)