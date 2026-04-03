import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

df = pd.read_csv("cases/Zhengzhou_Edgelist.csv")

node_cnt = max(max(df['START_NODE']), max(df['END_NODE']))
print(node_cnt)

neighbors = {}
for idx, row in df.iterrows():
    start = row['START_NODE']
    end = row['END_NODE']
    neighbors.setdefault(start, []).append(end)

# 度数
deg = [len(neighbors[node]) for node in neighbors]
max_deg=max(deg)
min_deg=min(deg)
print(f"min_deg: {min_deg}, max_deg: {max_deg}")
counter=Counter(deg)
for c in sorted(counter):
    print(f"deg{c}: {counter[c]}")

# 聚类系数
cluster_coef = {}
for cur, neigh_list in neighbors.items():
    n = len(neigh_list)
    if n < 2:
        cluster_coef[cur] = 0
        continue
    max_possible = n * (n - 1) / 2
    actual = 0
    neigh_set = set(neigh_list)
    for i in range(n):
        u = neigh_list[i]
        for j in range(i+1, n):
            v = neigh_list[j]
            if v in neighbors.get(u, []):
                actual += 1
    cluster_coef[cur] = actual / max_possible

cluster_coef_list=[val for idx,val in cluster_coef.items()]

bins = np.arange(0, 1, 0.01)
plt.hist(cluster_coef_list,bins=bins)
plt.xlabel("cluster_coef")
plt.ylabel("cnt")
plt.show()

# bins = np.arange(0, 10, 1)
# plt.hist(deg,bins=bins)
# plt.xlabel("deg")
# plt.ylabel("cnt")
# plt.show()