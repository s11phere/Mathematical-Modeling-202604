import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, deque
from q1 import dijkstra
import os

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


def process(df: pd.DataFrame, city_name=""):
    G = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        w = row['LENGTH']
        G.setdefault(u, []).append((v, w))

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

        comp_cnt += 1
    
    print(f"total components: {comp_cnt}")
    common_keys = set(d.keys()) & set(cnt.keys())
    sizes = [cnt[k] for k in common_keys]
    diams = [d[k] for k in common_keys]

    removed_cnt = 0
    for idx in range(comp_cnt):
        if sizes[idx] < 10 and diams[idx] < 5000:
            removed_cnt += 1
            for node in comps[idx]:
                removed.add(node)

    print(f"remove {removed_cnt} components, {len(removed)} nodes")

    mask = ~(df['START_NODE'].isin(removed) | df['END_NODE'].isin(removed))
    filtered_df = df[mask].copy()

    output_path = f"cases/{city_name}_filtered_edgelist.csv"
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered edge list saved to {output_path}")

def main():
    for filepath in city_files:
        city_name = os.path.basename(filepath).replace("_filtered_Edgelist.csv", "")
        print(f"\nCurrent City: {city_name}")
        df = pd.read_csv(filepath)
        process(df,city_name)

if __name__=="__main__":
    main()