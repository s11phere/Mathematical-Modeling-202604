import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import heapq
from collections import deque
import os
from q1 import build_graph

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

def dijkstra_with_prev(src, graph):
    """返回 (farthest_node, max_dist, prev_dict)"""
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
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
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    max_dist = max(dist.values())
    farthest = next(node for node, d in dist.items() if d == max_dist)
    return farthest, max_dist, prev

def dijkstra(src, graph):
    dist = {node: float('inf') for node in graph}
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

def compute_diameter(df: pd.DataFrame):
    """计算直径（2-近似），并返回直径值和路径节点列表"""
    G = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        w = row['LENGTH']
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

    max_graph = {node: G[node] for node in max_comp_nodes}
    print(f"total_nodes: {len(G)}, max_node_cnt: {max_size}, connected: {len(G)==max_size}")

    s = next(iter(max_graph))
    nd1, _ = dijkstra(s, max_graph)                # 第一次，得到一端
    nd2, diameter, prev = dijkstra_with_prev(nd1, max_graph)  # 第二次，记录前驱

    # 重建从 nd1 到 nd2 的路径
    path = []
    cur = nd2
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()   # 现在 path 从 nd1 到 nd2

    return diameter, path

def plot_network_with_path(csv_file, path_nodes, diameter=None, figsize=(12, 10), dpi=150, with_path=True):
    """绘制整个网络，并用红色高亮指定的路径"""
    df = pd.read_csv(csv_file)
    
    # 提取所有节点的坐标
    node_coords = {}
    for _, row in df.iterrows():
        node = row['START_NODE']
        x, y = row['XCoord'], row['YCoord']
        if node not in node_coords:
            node_coords[node] = (x, y)
    
    # 构建所有边的线段
    edges = []
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        if u in node_coords and v in node_coords:
            edges.append([(node_coords[u][0], node_coords[u][1]),
                          (node_coords[v][0], node_coords[v][1])])
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制所有边
    lc = LineCollection(edges, linewidths=0.5, colors='gray', alpha=0.3)
    ax.add_collection(lc)
    
    # 绘制所有节点
    xs = [c[0] for c in node_coords.values()]
    ys = [c[1] for c in node_coords.values()]
    ax.scatter(xs, ys, s=1, c='lightblue', alpha=0.6, edgecolors='none')
    
    # 高亮路径上的边和节点
    if with_path and path_nodes:
        # 获取路径上节点的坐标
        path_coords = [node_coords[node] for node in path_nodes if node in node_coords]
        if len(path_coords) >= 2:
            # 路径边
            path_edges = []
            for i in range(len(path_coords)-1):
                path_edges.append([path_coords[i], path_coords[i+1]])
            lc_path = LineCollection(path_edges, linewidths=2, colors='red', alpha=1)
            ax.add_collection(lc_path)
            # 路径节点
            path_x = [c[0] for c in path_coords]
            path_y = [c[1] for c in path_coords]
            ax.scatter(path_x, path_y, s=20, c='red', edgecolors='white', zorder=5)
    
    ax.set_aspect('equal')
    title = "Network with Diameter Path"
    if diameter is not None:
        title += f" (Length ≈ {diameter:.2f})"
    ax.set_title(title)
    ax.set_xlabel("XCoord")
    ax.set_ylabel("YCoord")
    plt.tight_layout()
    if not with_path:
        plt.savefig(f"figures/{city_name}_graph_with_edge.png", dpi=300)
    else:
        plt.savefig(f"figures/{city_name}_graph_with_edge_and_diam.png", dpi=300)

def plot_network_with_one_way_edges(csv_file, figsize=(12, 10), dpi=150):
    """绘制整个网络，并用红色高亮单向路径"""
    df = pd.read_csv(csv_file)
    
    # 提取所有节点的坐标
    node_coords = {}
    for _, row in df.iterrows():
        node = row['START_NODE']
        x, y = row['XCoord'], row['YCoord']
        if node not in node_coords:
            node_coords[node] = (x, y)
    
    edges = []
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        if u in node_coords and v in node_coords:
            edges.append([(node_coords[u][0], node_coords[u][1]),
                          (node_coords[v][0], node_coords[v][1])])
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制所有边
    lc = LineCollection(edges, linewidths=0.5, colors='gray', alpha=0.3)
    ax.add_collection(lc)

    directed_edges = []
    G = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        G.setdefault(u, []).append(v)
        
    for u,neighbor in G.items():
        for v in neighbor:
            if u not in G[v]:
                directed_edges.append([(node_coords[u][0], node_coords[u][1]),
                          (node_coords[v][0], node_coords[v][1])])    
    
    # 绘制所有单向边
    print(len(directed_edges))
    lc_path = LineCollection(directed_edges, linewidths=2, colors='red', alpha=1)
    ax.add_collection(lc_path)
    
    # 绘制所有节点
    xs = [c[0] for c in node_coords.values()]
    ys = [c[1] for c in node_coords.values()]
    ax.scatter(xs, ys, s=1, c='lightblue', alpha=0.6, edgecolors='none')

    ax.set_aspect('equal')
    title = "Network with One-Way Path"
    ax.set_title(title)
    ax.set_xlabel("XCoord")
    ax.set_ylabel("YCoord")
    plt.tight_layout()

    plt.show()
    

if __name__ == "__main__":
    for filepath in city_files:
        city_name = os.path.basename(filepath).replace("_Edgelist.csv", "")
        df = pd.read_csv(filepath)
        diameter, path = compute_diameter(df)
        print(f"近似直径长度: {diameter}")
        print(f"路径上的节点数: {len(path)}")
        # plot_network_with_path(filepath, path, diameter,with_path=False)
        # plot_network_with_path(filepath, path, diameter,with_path=True)
        plot_network_with_one_way_edges(filepath,dpi=300)
        