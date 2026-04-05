import pandas as pd
import networkx as nx
from q3_lir import get_data
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import os
from pathlib import Path
import random





def shed_k(G: nx.Graph,k):
    G_temp = G.copy()

    for nod in G.nodes():
        if nx.degree(G,nod) < k:
            G_temp.remove_node(nod)
    return(G_temp)







def throw(G, n):
    """
    移除图 G 中所有节点数 <= n 的连通分量。
    参数:
        G: networkx.Graph (无向图)
        n: int，阈值，保留节点数 > n 的分量
    返回:
        新图，仅包含节点数 > n 的连通分量
    """
    # 找出所有连通分量中节点数大于 n 的节点集合
    large_components = [comp for comp in nx.connected_components(G) if len(comp) > n]
    # 合并这些节点
    nodes_to_keep = set().union(*large_components) if large_components else set()
    # 返回子图
    return G.subgraph(nodes_to_keep).copy()


def down_town(G):
    G_temp = nx.k_core(G,k=2)
    G_temp = throw(G_temp,n=20)
    G_temp = shed_k(G_temp,k=3)
    G_temp = throw(G_temp,n=30)
    G_temp = nx.k_core(G_temp,k=2)
    G_temp = throw(G_temp,n=40)
    G_temp = shed_k(G_temp,k=3)
    G_temp = throw(G_temp,n=40)
    G_temp = nx.k_core(G_temp,k=2)
    G_temp = throw(G_temp,n=40)
    return(G_temp)



import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

def plot_network(G, figsize=(10, 8), node_size=5, edge_width=0.5, save_path=None, color_by_component=False):
    """
    绘制路网图（节点 + 道路）
    
    参数:
        G: networkx.Graph，节点需包含 'x' 和 'y' 坐标属性
        figsize: 图像尺寸，默认 (10,8)
        node_size: 节点大小，默认 5
        edge_width: 边线宽，默认 0.5
        save_path: 保存路径，若为 None 则显示图像
        color_by_component: 是否按连通分量着色，默认 False（统一蓝色节点、灰色边）
    """
    # 提取节点坐标
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    if not color_by_component:
        # 原有逻辑：统一颜色
        lines = []
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            lines.append([(x1, y1), (x2, y2)])
        
        lc = LineCollection(lines, linewidths=edge_width, colors='gray', alpha=0.7)
        ax.add_collection(lc)
        
        x_vals = [pos[node][0] for node in G.nodes]
        y_vals = [pos[node][1] for node in G.nodes]
        ax.scatter(x_vals, y_vals, s=node_size, c='blue', alpha=0.8, edgecolors='none')
    else:
        # 按连通分量着色
        components = list(nx.connected_components(G))
        # 为每个分量分配一个颜色（使用 colormap，分量多时可循环）
        cmap = plt.cm.get_cmap('tab20')  # 最多20种不同颜色，超出会循环
        colors = [cmap(i % 20) for i in range(len(components))]
        
        for comp_idx, comp_nodes in enumerate(components):
            subG = G.subgraph(comp_nodes)
            # 获取该分量的边线段
            lines = []
            for u, v in subG.edges():
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                lines.append([(x1, y1), (x2, y2)])
            if lines:
                lc = LineCollection(lines, linewidths=edge_width, colors=[colors[comp_idx]], alpha=0.7)
                ax.add_collection(lc)
            
            # 绘制该分量的节点
            x_vals = [pos[node][0] for node in comp_nodes]
            y_vals = [pos[node][1] for node in comp_nodes]
            ax.scatter(x_vals, y_vals, s=node_size, c=[colors[comp_idx]], alpha=0.8, edgecolors='none')
    
    # 设置坐标轴等比例，并自动调整范围
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('X cord')
    ax.set_ylabel('Y cord')
    ax.set_title('city_map')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.show()
    
    return fig, ax





def get_capacity_network(G):
    # 给所有边添加容量属性,并实现单点分离
    for u in G.nodes():
        for v in list(G[u].keys()):
            G[u][v]['capacity'] = 1
    
    # for u in G.nodes():
        
    return(G)














if __name__ == "__main__":
    city_names = ["Chengdu","Dalian","Dongguan","Harbin","Qingdao","Quanzhou","Shenyang","Zhengzhou"]
    file_dir = Path(__file__).parent
    save_path = file_dir / "q5_demo_plot"



    for city_name in city_names:
        if city_name != "Qingdao":
            continue

        img_path = file_dir / "q5_demo_plot" / f"{city_name}_network.png"


        G = get_data(city_name)

        print(type(G.nodes()))

        u = random.choice(list(G.nodes()))
        print(u)
        print(type(u))
        print(G[u])
        print(type(G[u]))

        G = get_capacity_network(G)

        u = random.choice(list(G.nodes()))
        print(G[u])
        

        