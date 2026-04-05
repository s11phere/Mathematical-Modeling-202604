import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys

def create_grid_graph(n):
    """
    创建 n x n 的正方形格子图（无向图），节点总数为 n^2。
    """
    G = nx.grid_2d_graph(n, n)
    # 将节点重命名为 0 到 n^2-1 的整数，便于处理
    G = nx.convert_node_labels_to_integers(G)
    return G

def remove_edges_randomly(G, p):
    """
    随机移除比例为 p 的边。
    p: 移除边数占总边数的比例 (0 <= p <= 1)
    """
    edges = list(G.edges())
    num_remove = int(len(edges) * p)
    # 随机选择要移除的边
    remove_edges = np.random.choice(len(edges), size=num_remove, replace=False)
    edges_to_remove = [edges[i] for i in remove_edges]
    G_removed = G.copy()
    G_removed.remove_edges_from(edges_to_remove)
    return G_removed

def plot_degree_distribution(G, p, savefig=None):
    """
    绘制度数分布的直方图，并标注理论预期（可选）。
    """
    degrees = [d for n, d in G.degree()]
    degree_counts = np.bincount(degrees)
    degree_vals = np.arange(len(degree_counts))
    # 只显示度数 >0 的部分
    mask = degree_counts > 0
    degree_vals = degree_vals[mask]
    degree_counts = degree_counts[mask]

    plt.figure(figsize=(10, 6))
    plt.bar(degree_vals, degree_counts, width=0.8, alpha=0.7, color='steelblue',
            edgecolor='black', label='Observed')
    plt.xlabel('Degree (k)')
    plt.ylabel('Number of nodes')
    plt.title(f'Degree Distribution after removing {p*100:.1f}% of edges\n'
              f'Grid size: {int(np.sqrt(G.number_of_nodes()))}x{int(np.sqrt(G.number_of_nodes()))}, '
              f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
    plt.axvline(avg_deg, color='red', linestyle='--', linewidth=1.5,
                label=f'Average degree = {avg_deg:.2f}')
    plt.legend()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    n = 60
    if len(sys.argv) > 1:
        try:
            p = float(sys.argv[1])
        except ValueError:
            print("请输入一个有效的浮点数作为 p (0 <= p <= 1)。")
            return
    else:
        p = input("请输入要移除的边的比例 p (0~1): ")
        try:
            p = float(p)
        except ValueError:
            print("无效输入，请运行程序并输入一个数字。")
            return

    if p < 0 or p > 1:
        print("p 必须在 0 到 1 之间。")
        return

    print(f"正在创建 {n}x{n} 的格子网络...")
    G = create_grid_graph(n)
    print(f"原始网络: 节点数 = {G.number_of_nodes()}, 边数 = {G.number_of_edges()}")

    print(f"随机移除 {p*100:.1f}% 的边...")
    G_removed = remove_edges_randomly(G, p)
    print(f"剩余网络: 节点数 = {G_removed.number_of_nodes()}, 边数 = {G_removed.number_of_edges()}")

    # 计算度数分布
    degrees = [d for n, d in G_removed.degree()]
    print(f"度数统计: 最小度数 = {min(degrees)}, 最大度数 = {max(degrees)}, "
          f"平均度数 = {np.mean(degrees):.2f}")

    # 绘制分布图
    plot_degree_distribution(G_removed, p, savefig=f"degree_dist_p_{p:.2f}.png")

if __name__ == "__main__":
    main()