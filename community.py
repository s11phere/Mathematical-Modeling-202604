import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import numpy as np
import os
from q1 import degree_dist

city_files = [
    "cases/Chengdu_filtered_Edgelist.csv",
    "cases/Dalian_filtered_Edgelist.csv",
    "cases/Dongguan_filtered_Edgelist.csv",
    "cases/Harbin_filtered_Edgelist.csv",
    "cases/Qingdao_filtered_Edgelist.csv",
    "cases/Quanzhou_filtered_Edgelist.csv",
    "cases/Shenyang_filtered_Edgelist.csv",
    "cases/Zhengzhou_filtered_Edgelist.csv"
]

def build_graph_with_coords(df):
    """
    从 DataFrame 构建有向图（邻居列表去重）并返回节点坐标字典。
    """
    graph = {}
    coords = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        x = row['XCoord']
        y = row['YCoord']
        coords[u] = (x, y)
        graph.setdefault(u, []).append(v)
    for node in graph:
        graph[node] = list(set(graph[node]))

    return graph, coords

def count_nodes_in_radius(coords, center_node, radius):
    """
    计算以 center_node 为中心、半径为 radius 的圆内的节点数（欧氏距离）。
    使用暴力 O(N^2)，适用于中等规模网络。
    """
    cx, cy = coords[center_node]
    cnt = 0
    for node, (x, y) in coords.items():
        if (x - cx)**2 + (y - cy)**2 <= radius**2:
            cnt += 1
    return cnt

def find_center_by_density(coords, radius):
    """
    找出半径内节点数最多的中心节点。
    返回 (center_node, max_count)
    """
    best_node = None
    best_count = -1
    for node in coords:
        cnt = count_nodes_in_radius(coords, node, radius)
        if cnt > best_count:
            best_count = cnt
            best_node = node
    return best_node, best_count

def extract_subgraph_by_radius(graph, coords, center_node, radius):
    """
    提取以 center_node 为中心、半径 radius 内的子图。
    返回子图（字典形式，邻居列表去重）以及该区域内的节点集合。
    """
    # 1. 找出半径内的所有节点
    cx, cy = coords[center_node]
    nodes_in_radius = set()
    for node, (x, y) in coords.items():
        if (x - cx)**2 + (y - cy)**2 <= radius**2:
            nodes_in_radius.add(node)
    
    # 2. 构建子图：只保留两端都在 nodes_in_radius 中的边
    subgraph = {}
    for u, neighbors in graph.items():
        if u not in nodes_in_radius:
            continue
        filtered = [v for v in neighbors if v in nodes_in_radius]
        if filtered:
            subgraph[u] = list(set(filtered))
    return subgraph, nodes_in_radius

def process_city_center(csv_path, radius, output_csv=None):
    """
    主函数：从 CSV 读取数据，找出密度最高的中心区域，提取子图。
    
    参数:
        csv_path: 输入的 CSV 文件路径
        radius: 圆形区域半径（与坐标单位一致，如米）
        output_csv: 可选，将子图的边保存为 CSV 文件路径
    
    返回:
        subgraph: 子图字典 {node: [neighbors]}
        center_node: 中心节点
        nodes_in_radius: 半径内节点集合
    """
    df = pd.read_csv(csv_path)
    graph, coords = build_graph_with_coords(df)
    print(f"原始图: {len(graph)} 个节点, 总边数: {sum(len(nbr) for nbr in graph.values())}")
    
    center_node, max_count = find_center_by_density(coords, radius)
    print(f"中心节点: {center_node}, 半径内节点数: {max_count}")
    
    subgraph, nodes_set = extract_subgraph_by_radius(graph, coords, center_node, radius)
    print(f"子图: {len(subgraph)} 个节点, 边数: {sum(len(nbr) for nbr in subgraph.values())}")

    if output_csv:
        # 需要从原始 df 中过滤出两端都在 nodes_set 中的边
        mask = df['START_NODE'].isin(nodes_set) & df['END_NODE'].isin(nodes_set)
        filtered_df = df[mask].copy()
        filtered_df.to_csv(output_csv, index=False)
        print(f"子图边已保存到 {output_csv}")
    
    return subgraph, center_node, nodes_set

def plot_network_with_circle(graph, coords, center_node, radius, 
                             highlight_inside=True, circle_alpha=0.3,
                             figsize=(12, 10), dpi=300):
    """
    绘制整个网络，并用圆形区域标记以 center_node 为中心、半径为 radius 的区域。
    
    参数:
        graph: dict, {node: [neighbor1, neighbor2, ...]} 有向或无向（绘图时视为无向）
        coords: dict, {node: (x, y)}
        center_node: 中心节点
        radius: 圆形半径（与坐标单位一致）
        highlight_inside: 是否高亮圆形区域内的节点和边（True: 区域外灰色，区域内彩色）
        circle_alpha: 圆形填充透明度
        figsize, dpi: 图形尺寸和分辨率
    """
    # 计算圆形内的节点集合
    cx, cy = coords[center_node]
    nodes_inside = set()
    for node, (x, y) in coords.items():
        if (x - cx)**2 + (y - cy)**2 <= radius**2:
            nodes_inside.add(node)
    
    # 准备所有节点坐标和边线段
    all_nodes = list(coords.keys())
    xs = [coords[n][0] for n in all_nodes]
    ys = [coords[n][1] for n in all_nodes]
    
    # 构建所有边的线段（无向，每条边只绘制一次，避免重复）
    edges_set = set()
    edges = []
    for u, neighbors in graph.items():
        for v in neighbors:
            if (u, v) not in edges_set and (v, u) not in edges_set:
                edges_set.add((u, v))
                if u in coords and v in coords:
                    edges.append([coords[u], coords[v]])
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if highlight_inside:
        # 1. 绘制区域外的边（灰色半透明）
        lc_all = LineCollection(edges, linewidths=0.2, colors='lightgray', alpha=0.5)
        ax.add_collection(lc_all)
        
        # 2. 绘制区域内的边（深色高亮）
        inside_edges = []
        for u, neighbors in graph.items():
            if u not in nodes_inside:
                continue
            for v in neighbors:
                if v in nodes_inside:
                    # 避免重复绘制同一条边（无向）
                    if (u, v) not in edges_set:  # 简化，实际可以用 visited set
                        inside_edges.append([coords[u], coords[v]])
                        edges_set.add((u, v))
        lc_inside = LineCollection(inside_edges, linewidths=0.2, colors='red', alpha=0.8)
        ax.add_collection(lc_inside)
        
        # 3. 绘制节点：区域外灰色，区域内蓝色
        outside_nodes_x = [coords[n][0] for n in all_nodes if n not in nodes_inside]
        outside_nodes_y = [coords[n][1] for n in all_nodes if n not in nodes_inside]
        inside_nodes_x = [coords[n][0] for n in nodes_inside]
        inside_nodes_y = [coords[n][1] for n in nodes_inside]
        
        ax.scatter(outside_nodes_x, outside_nodes_y, s=0.5, c='gray', alpha=0.5, label='Outside')
        ax.scatter(inside_nodes_x, inside_nodes_y, s=0.5, c='blue', alpha=0.8, label='Inside')
    else:
        # 不高亮，只画所有边和所有节点，然后叠加圆形
        lc_all = LineCollection(edges, linewidths=0.5, colors='gray', alpha=0.3)
        ax.add_collection(lc_all)
        ax.scatter(xs, ys, s=0.5, c='lightblue', alpha=0.6)
    
    # 绘制圆形区域
    circle = Circle((cx, cy), radius, edgecolor='red', facecolor='none', linewidth=2, linestyle='--')
    # 若要填充半透明红色：facecolor='red', alpha=circle_alpha
    circle = Circle((cx, cy), radius, edgecolor='red', facecolor='red', alpha=circle_alpha)
    ax.add_patch(circle)
    
    # 标记中心节点
    ax.scatter(cx, cy, s=50, c='red', marker='*', edgecolors='black', label='Center')
    
    ax.set_aspect('equal')
    ax.set_title(f"Network with City Center (radius={radius})")
    ax.set_xlabel("XCoord")
    ax.set_ylabel("YCoord")
    ax.legend()
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    for filepath in city_files:
        df=pd.read_csv(filepath)
        city_name = os.path.basename(filepath).replace("_filtered_Edgelist.csv", "")
        graph, coords = build_graph_with_coords(df)
        radius = 20000
        center_node, _ = find_center_by_density(coords, radius)
        plot_network_with_circle(graph, coords, center_node, radius, highlight_inside=True)
        subgraph,_=extract_subgraph_by_radius(graph, coords, center_node, radius)
        degree_dist(subgraph,city_name,upper_bound=5)