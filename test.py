import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection
from collections import defaultdict

# ========== 图构建和最大连通分量函数 ==========
def build_graph(df):
    """从边列表DataFrame构建无向图（邻接表字典）"""
    G = defaultdict(set)
    for _, row in df.iterrows():
        u, v = row['START_NODE'], row['END_NODE']
        G[u].add(v)
        G[v].add(u)
    return {k: list(v) for k, v in G.items()}

def max_link(G):
    """返回当前图的最大连通分量大小（节点数）"""
    visited = set()
    max_size = 0
    for node in G:
        if node not in visited:
            queue = [node]
            visited.add(node)
            size = 0
            while queue:
                cur = queue.pop()
                size += 1
                for nb in G[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if size > max_size:
                max_size = size
    return max_size

def load_node_coordinates(filepath):
    """从CSV文件中读取每个节点的坐标（XCoord, YCoord）"""
    df = pd.read_csv(filepath)
    coords = {}
    for _, row in df.iterrows():
        start = row['START_NODE']
        end = row['END_NODE']
        x = row['XCoord']
        y = row['YCoord']
        if start not in coords:
            coords[start] = (x, y)
        if end not in coords:
            coords[end] = (x, y)
    return coords

def read_delete_sequence_from_file(filepath):
    """
    从文件读取删除节点序列。
    支持格式：
    - 每行一个节点编号
    - 一行内空格/逗号分隔多个编号
    - 空行自动跳过
    """
    nodes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            for p in parts:
                try:
                    nodes.append(int(p))
                except ValueError:
                    print(f"警告：忽略无效数字 '{p}'")
    return nodes

def plot_and_show(G, coords, removed_set, step, comp_size):
    """
    绘制当前图状态：
    1. 先绘制所有节点和边（淡色背景）
    2. 再绘制未删除的节点和边（亮色前景）
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    all_nodes = set(coords.keys())
    
    # 第一步：所有边和节点（背景，淡色）
    all_edges = []
    for u in G:
        for v in G[u]:
            if u < v:
                all_edges.append((u, v))
    if all_edges:
        segs = [[coords[u], coords[v]] for u, v in all_edges]
        lc_all = LineCollection(segs, colors='lightgray', linewidths=0.5, alpha=0.2)
        ax.add_collection(lc_all)
    
    xs_all = [coords[n][0] for n in all_nodes]
    ys_all = [coords[n][1] for n in all_nodes]
    ax.scatter(xs_all, ys_all, s=20, c='lightgray', alpha=0.3, edgecolors='none')
    
    # 第二步：未删除的节点和边（前景，亮色）
    active_edges = []
    for u in G:
        if u in removed_set:
            continue
        for v in G[u]:
            if v not in removed_set and u < v:
                active_edges.append((u, v))
    if active_edges:
        segs_active = [[coords[u], coords[v]] for u, v in active_edges]
        lc_active = LineCollection(segs_active, colors='blue', linewidths=1.2, alpha=0.4)
        ax.add_collection(lc_active)
    
    active_nodes = [n for n in all_nodes if n not in removed_set and n in G]
    if active_nodes:
        xs_active = [coords[n][0] for n in active_nodes]
        ys_active = [coords[n][1] for n in active_nodes]
        ax.scatter(xs_active, ys_active, s=5, c='red', alpha=0.8, zorder=3)
    
    deleted_nodes = [n for n in removed_set if n in coords]
    if deleted_nodes:
        xs_del = [coords[n][0] for n in deleted_nodes]
        ys_del = [coords[n][1] for n in deleted_nodes]
        ax.scatter(xs_del, ys_del, c='red', s=5, linewidths=2, zorder=4)
    
    ax.set_title(f"Step {step} | Removed {len(removed_set)} nodes | Largest Component Size: {comp_size}")
    ax.axis('equal')
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

def simulate_deletion_with_sequence(delete_sequence, G, coords, update_freq=5):
    removed = set()
    step = 0
    total_steps = len(delete_sequence)
    history = []  # 记录每一步的 (step, comp_size)
    
    # 初始状态（第0步）
    init_comp = max_link(G)
    history.append((0, init_comp))
    plot_and_show(G, coords, set(), 0, init_comp)
    print("初始地图已显示，关闭窗口后开始删除节点...")
    
    for node in delete_sequence:
        if node not in G:
            print(f"警告：节点 {node} 不在图中，已跳过")
            continue
        # 删除节点
        for nbr in G[node]:
            G[nbr].remove(node)
        del G[node]
        removed.add(node)
        step += 1
        comp_size = max_link(G)
        history.append((step, comp_size))
        
        if step % update_freq == 0 or step == total_steps:
            print(f"Step {step}: largest component = {comp_size}")
            plot_and_show(G, coords, removed, step, comp_size)
        elif step % 100 == 0:
            print(f"处理中... 已删除 {step} / {total_steps} 个节点")
    
    print(f"模拟完成，共删除 {len(removed)} 个节点。")
    return history

def main():
    city_csv = "cases/Chengdu_filtered_Edgelist.csv"
    print("正在加载成都地图数据...")
    df = pd.read_csv(city_csv)
    G = build_graph(df)
    coords = load_node_coordinates(city_csv)
    node_cnt = len(G)
    print(f"图中共有 {node_cnt} 个节点。")
    
    delete_seq = read_delete_sequence_from_file("data.txt")
    print(f"从文件读取到 {len(delete_seq)} 个节点: {delete_seq[:10]}{'...' if len(delete_seq)>10 else ''}")
    
    history = simulate_deletion_with_sequence(delete_seq, G, coords, update_freq=100)
    
    print(sum(x for y,x in history)/(node_cnt*node_cnt))
    # 保存历史数据到CSV
    df_history = pd.DataFrame(history, columns=['step', 'component_size'])
    df_history.to_csv('deletion_history.csv', index=False)
    print("已保存每一步的最大连通分量到 deletion_history.csv")
    
    # 绘制变化曲线
    steps = [h[0] for h in history]
    sizes = [h[1] for h in history]
    plt.figure(figsize=(10, 6))
    plt.plot(steps, sizes, marker='.', linestyle='-', linewidth=1, markersize=3)
    plt.xlabel('Number of Deleted Nodes')
    plt.ylabel('Largest Component Size')
    plt.title('Evolution of Largest Component Size during Node Deletion')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('component_size_evolution.png', dpi=150)
    plt.show()
    print("已保存变化曲线图 component_size_evolution.png")

if __name__ == "__main__":
    main()