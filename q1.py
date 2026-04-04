import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, deque
from scipy import stats
import os
from matplotlib.collections import LineCollection
import heapq
import powerlaw
import powerlaw
import matplotlib.ticker as ticker

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


def get_processed_graph(df: pd.DataFrame, city_name="", plot=False, include_length=False):
    G = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        w = row['LENGTH']
        G.setdefault(u, []).append((v, w))

    node_component = {}
    comps = []
    d = {}
    cnt = {}
    visited = set()
    removed = set()
    comp_cnt = 0

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

        # 为分量内所有节点分配相同的分量 ID
        for n in comp_nodes:
            node_component[n] = comp_cnt

        subgraph = {n: G[n] for n in comp_nodes}
        if len(subgraph) > 1:
            s = next(iter(subgraph))
            nd1, _ = dijkstra(s, subgraph)
            _, diameter = dijkstra(nd1, subgraph)
        else:
            diameter = 0
        d[comp_cnt] = diameter
        cnt[comp_cnt] = len(comp_nodes)
        comps.append(comp_nodes)
        # print(f"component {comp_cnt}, nodes: {len(comp_nodes)}, diameter: {diameter}")

        comp_cnt += 1
    
    print(f"total components: {comp_cnt}")
    common_keys = set(d.keys()) & set(cnt.keys())
    sizes = [cnt[k] for k in common_keys]
    diams = [d[k] for k in common_keys]

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(sizes, diams, s=3, alpha=0.6)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of nodes (log scale)')
        plt.ylabel('Diameter (log scale)')
        plt.title(
            f'Component Size vs Diameter - {city_name} - {comp_cnt} components')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"figures/{city_name}_size_diam_features.png", dpi=150)

    removed_cnt = 0
    for idx in range(comp_cnt):
        if sizes[idx] < 10 and diams[idx] < 5000:
            removed_cnt += 1
            for node in comps[idx]:
                removed.add(node)

    print(f"remove {removed_cnt} components, {len(removed)} nodes")

    new_G = {}
    for _, row in df.iterrows():
        u = row['START_NODE'].astype(np.int64)
        v = row['END_NODE'].astype(np.int64)
        w = row['LENGTH']
        if u not in removed and v not in removed:
            if include_length:
                new_G.setdefault(u, []).append((v, w))
            else:
                new_G.setdefault(u, []).append(v)
    return new_G, node_component


def plot_components(df, G, node_component):
    # 提取所有节点的坐标
    node_coords = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        x, y = row['XCoord'], row['YCoord']
        if u in G:
            node_coords.setdefault(u, (x, y))
        if v in G:
            node_coords.setdefault(v, (x, y))

    # 所有边
    edges = []
    for u, neighbors in G.items():
        for v, _ in neighbors:
            if u in node_coords and v in node_coords:
                edges.append([node_coords[u], node_coords[v]])

    # 为每个节点获取分量 ID
    colors = [node_component[node]
              for node in node_coords.keys() if node in node_component]

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

    # 绘制所有边（灰色半透明）
    if edges:
        lc = LineCollection(edges, linewidths=0.5, colors='gray', alpha=0.3)
        ax.add_collection(lc)

    # 绘制节点（按分量着色）
    xs = [node_coords[node][0]
          for node in node_coords if node in node_component]
    ys = [node_coords[node][1]
          for node in node_coords if node in node_component]
    sc = ax.scatter(xs, ys, s=5, c=colors, cmap='tab20',
                    alpha=0.8, edgecolors='none')

    ax.set_aspect('equal')
    ax.set_title("Network Components with Different Colors")
    ax.set_xlabel("XCoord")
    ax.set_ylabel("YCoord")
    plt.tight_layout()
    plt.show()


def build_graph(df: pd.DataFrame):
    """建立有向图"""
    graph = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        graph.setdefault(u, []).append(v)
    for node in graph:
        graph[node] = list(set(graph[node]))
    return graph

def degree_dist(G: dict, city_name,plot_fit=True, upper_bound=None):
    """
    计算度序列及度分布频数，并用多种分布拟合（对数正态、指数）。
    """
    deg = [len(neigh) for neigh in G.values()]
    deg_counter = Counter(deg)
    fit_summary = {}
    
    # 根据 upper_bound 过滤度序列（用于拟合）
    deg_array_full = np.array([d for d in deg if d > 0])   # 去掉零度
    if upper_bound is not None:
        deg_array = deg_array_full[deg_array_full <= upper_bound]
        print(f"拟合时使用度数 <= {upper_bound} 的数据点，共 {len(deg_array)} 个（原始正度数 {len(deg_array_full)} 个）")
    else:
        deg_array = deg_array_full
        print(f"拟合时使用所有正度数，共 {len(deg_array)} 个")
    
    # 1. 对数正态分布拟合
    shape, loc, scale = stats.lognorm.fit(deg_array, floc=0)
    loglik_ln = np.sum(stats.lognorm.logpdf(deg_array, shape, loc=0, scale=scale))
    fit_summary['lognormal'] = {
        'params': (shape, scale),
        'log_likelihood': loglik_ln,
        'description': f'Log-normal (σ={shape:.3f}, μ={np.log(scale):.3f})'
    }
    
    # 2. 指数分布拟合
    loc_exp, scale_exp = stats.expon.fit(deg_array, floc=0)
    loglik_exp = np.sum(stats.expon.logpdf(deg_array, loc=0, scale=scale_exp))
    fit_summary['exponential'] = {
        'params': (scale_exp,),
        'log_likelihood': loglik_exp,
        'description': f'Exponential (λ={1/scale_exp:.3f})'
    }
    
    # 输出拟合对比
    print("\n=== 分布拟合对比===")
    valid_fits = {k: v for k, v in fit_summary.items() if v is not None}
    sorted_fits = sorted(valid_fits.items(), key=lambda x: x[1]['log_likelihood'], reverse=True)
    for name, info in sorted_fits:
        print(f"{name:12s}: {info['log_likelihood']:.2f}  ({info['description']})")
    
    # 绘图
    if plot_fit:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # 计算经验 CCDF（基于原始所有数据）
        unique_deg = sorted(set(deg))
        n_total = len(deg)
        ccdf = [sum(1 for d in deg if d >= k) / n_total for k in unique_deg]
        ax.loglog(unique_deg, ccdf, 'ko', markersize=4, alpha=0.5, label='Empirical (all data)')
        
        # 生成 x 轴（用于绘制拟合曲线，覆盖拟合数据范围即可）
        x_max_for_fit = upper_bound if upper_bound is not None else max(unique_deg)
        x_vals = np.logspace(np.log10(min(unique_deg)), np.log10(max(unique_deg)), 200)
        
        colors = {'lognormal': 'red', 'exponential': 'green', 'powerlaw': 'blue'}
        linestyles = {'lognormal': '-', 'exponential': ':', 'powerlaw': '--'}
        
        # 对数正态 CCDF
        if fit_summary.get('lognormal'):
            shape, scale = fit_summary['lognormal']['params']
            ccdf_ln = stats.lognorm.sf(x_vals, shape, loc=0, scale=scale)
            ax.loglog(x_vals, ccdf_ln, color=colors['lognormal'], linestyle=linestyles['lognormal'],
                      label=fit_summary['lognormal']['description'])
        
        # 指数 CCDF
        if fit_summary.get('exponential'):
            scale_exp = fit_summary['exponential']['params'][0]
            ccdf_exp = stats.expon.sf(x_vals, loc=0, scale=scale_exp)
            ax.loglog(x_vals, ccdf_exp, color=colors['exponential'], linestyle=linestyles['exponential'],
                      label=fit_summary['exponential']['description'])
        
        if upper_bound is not None:
            ax.axvline(x=upper_bound, color='gray', linestyle='-.', alpha=0.5, label=f'Upper bound = {upper_bound}')
        
        ax.set_xlabel('Degree (k)')
        ax.set_ylabel('P(K ≥ k)')
        ax.legend()
        
        plt.title(f"degree distribution fitting for {city_name}")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='both')
        
        plt.tight_layout()
        plt.savefig(f"figures/{city_name}_deg_dist_fit.png", dpi=150)
    
    return deg, deg_counter, fit_summary

def cluster(G: dict):
    """计算平均聚类系数，G 的值为 (邻居, 权重) 列表"""
    cluster_coef = {}
    for cur, neighbor in G.items():
        # 提取邻居节点 ID（忽略权重）
        nbr_ids = [nbr for nbr, _ in neighbor]
        n = len(nbr_ids)
        if n < 2:
            cluster_coef[cur] = 0
            continue
        max_possible = n * (n - 1) / 2
        actual = 0
        nbr_set = set(nbr_ids)
        for i in range(n):
            u = nbr_ids[i]
            for j in range(i+1, n):
                v = nbr_ids[j]
                if any(nbr == v for nbr, _ in G.get(u, [])):
                    actual += 1
        cluster_coef[cur] = actual / max_possible
    avg_c = np.mean(list(cluster_coef.values())) if cluster_coef else 0
    return avg_c, cluster_coef


def dijkstra(src, graph):
    """返回 (dist_list, farthest_node, max_dist)"""
    n = len(graph)
    dist = {}
    for node in graph:
        dist[node] = float('inf')
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
    """计算直径，考虑最大连通分量"""
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

    # 构建最大连通分量的子图
    max_graph = {node: G[node] for node in max_comp_nodes}
    print(
        f"total_nodes: {len(G)}, max_node_cnt: {max_size}, connected: {len(G)==max_size}")

    # 计算直径
    s = next(iter(max_graph))
    nd1, _ = dijkstra(s, max_graph)
    _, diameter = dijkstra(nd1, max_graph)

    return diameter

def main():
    for filepath in city_files:
        city_name = os.path.basename(filepath).replace("_Edgelist.csv", "")
        print(f"\nCurrent City: {city_name}")
        df = pd.read_csv(filepath)

        # 构建无向邻居字典
        G, node_comp = get_processed_graph(
            df, city_name, plot=True, include_length=True)
        node_cnt = len(G)
        print(f"  节点数: {node_cnt}")

        # 度分布
        deg_seq, deg_cnt, res = degree_dist(G,city_name,upper_bound=5)
        max_deg = max(deg_seq)
        min_deg = min(deg_seq)
        print(f"  最小度: {min_deg}, 最大度: {max_deg}")
        print("  度分布:")
        for d in sorted(deg_cnt):
            print(f"    deg {d}: {deg_cnt[d]}")

        # 聚类系数
        avg_c, cluster_coef_dict = cluster(G)
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

# def main():
#     for filepath in city_files:
#         city_name = os.path.basename(filepath).replace("_Edgelist.csv", "")
#         print(f"\nCurrent City: {city_name}")
#         df = pd.read_csv(filepath)

#         # plot_components(df,G,node_comp)


if __name__ == "__main__":
    main()
