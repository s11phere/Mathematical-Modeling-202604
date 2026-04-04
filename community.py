import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt

# ==================== 数据加载 ====================
df = pd.read_csv("cases/Quanzhou_Edgelist.csv")

# 构建 NetworkX 图（无向，带权重）
G_nx = nx.Graph()
for _, row in df.iterrows():
    u = row['START_NODE']
    v = row['END_NODE']
    w = row['LENGTH']
    G_nx.add_edge(u, v, weight=w)

print(f"节点数: {G_nx.number_of_nodes()}")
print(f"边数: {G_nx.number_of_edges()}")

# ==================== 转换为 igraph ====================
# 注意：igraph 会自动将节点映射为 0..N-1 的索引，原始 ID 保存在 '_nx_name' 属性中
G_ig = ig.Graph.from_networkx(G_nx)

# ==================== Leiden 社区发现 ====================
partition = la.find_partition(
    G_ig,
    la.CPMVertexPartition,
    resolution_parameter=0.5,   # 可调整，值越小社区越少
    weights='weight',
    n_iterations=2,
    seed=42
)

# 正确获取节点到社区的映射：使用原始节点 ID
node_communities = {}
for comm_id, community in enumerate(partition):
    for node_idx in community:
        original_node = G_ig.vs[node_idx]["_nx_name"]   # 提取原始 ID
        node_communities[original_node] = comm_id

print(f"检测到的社区数量: {len(partition)}")
sizes = [len(c) for c in partition]
print(f"社区大小: 最小={min(sizes)}, 最大={max(sizes)}, 平均={sum(sizes)/len(sizes):.1f}")

# 可选：将社区标签添加到 NetworkX 节点属性
nx.set_node_attributes(G_nx, node_communities, 'community')

# ==================== 坐标提取与可视化 ====================
# 提取节点坐标（确保每个节点都能找到坐标）
node_coords = {}
for _, row in df.iterrows():
    node_coords[row['START_NODE']] = (row['XCoord'], row['YCoord'])
# 对于只出现在 END_NODE 的节点，尝试从 END_NODE 行获取坐标
for _, row in df.iterrows():
    node_coords.setdefault(row['END_NODE'], (row['XCoord'], row['YCoord']))

# 构建位置字典，缺失坐标的节点使用 (0,0) 并给出警告
pos = {}
missing_coords = []
for node in G_nx.nodes():
    if node in node_coords:
        pos[node] = node_coords[node]
    else:
        pos[node] = (0, 0)
        missing_coords.append(node)
if missing_coords:
    print(f"警告：以下节点缺少坐标，已设为 (0,0)：{missing_coords[:10]}...")

# 绘图
plt.figure(figsize=(14, 12))
colors = [node_communities[node] for node in G_nx.nodes()]

nx.draw_networkx_nodes(G_nx, pos, node_size=5, node_color=colors,
                       cmap='tab20', alpha=0.8, edgecolors='none')
nx.draw_networkx_edges(G_nx, pos, alpha=0.1, width=0.5, edge_color='gray')
plt.title(f"Leiden 社区发现 (resolution=0.5) — {len(partition)} 个社区")
plt.axis('equal')
plt.tight_layout()
plt.savefig("leiden_communities.png", dpi=300)
plt.show()