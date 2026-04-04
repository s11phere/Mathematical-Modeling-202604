import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from collections import Counter

def create_scale_free_network(num_nodes, m):
    """
    生成无标度网络（BA模型）
    :param num_nodes: 总节点数
    :param m: 每个新节点添加的边数（初始网络为 m 个节点的完全图）
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    # 初始网络：m 个节点的完全图
    G = nx.complete_graph(m)
    # 如果总节点数小于初始节点数，直接返回
    if num_nodes <= m:
        return G
    
    # 逐步添加新节点
    for new_node in range(m, num_nodes):
        G.add_node(new_node)
        targets = preferential_attachment(G, m)
        G.add_edges_from((new_node, target) for target in targets)
    return G

def preferential_attachment(G, num_edges):
    """基于度优先选择目标节点（无放回）"""
    # 获取所有节点的度数列表
    nodes = list(G.nodes())
    degrees = np.array([G.degree(n) for n in nodes])
    # 处理全零度的情况（理论上不会发生，因为初始网络有边）
    if degrees.sum() == 0:
        # 如果所有节点度数为0（孤立节点），则均匀随机选择
        probs = np.ones(len(nodes)) / len(nodes)
    else:
        probs = degrees / degrees.sum()
    # 按概率选择目标节点
    return np.random.choice(nodes, size=num_edges, replace=False, p=probs)

def plot_degree_distribution(G, fit_powerlaw=True):
    degrees = [d for n, d in G.degree()]
    deg_counts = Counter(degrees)
    sorted_deg = sorted(deg_counts.keys())
    freq = [deg_counts[d] for d in sorted_deg]
    
    # 直方图（线性坐标）
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(sorted_deg, freq, width=0.8, color='skyblue')
    plt.xlim((0,20))
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution (linear)')
    plt.grid(True, alpha=0.3)
    
    # CCDF（双对数坐标）
    plt.subplot(1, 2, 2)
    total = G.number_of_nodes()
    ccdf = [sum(1 for d in degrees if d >= k) / total for k in sorted_deg]
    plt.loglog(sorted_deg, ccdf, 'bo', markersize=4)
    plt.xlabel('Degree (k)')
    plt.ylabel('P(K ≥ k)')
    plt.title('CCDF (log-log)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if fit_powerlaw:
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        print(f"Power-law exponent γ = {fit.power_law.alpha:.3f}, xmin = {fit.power_law.xmin}")
        return fit

def main():
    G = create_scale_free_network(num_nodes=1000, m=1)
    plot_degree_distribution(G)

if __name__ == "__main__":
    main()