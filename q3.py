import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection
import random
from q1 import build_graph
from q2 import max_link

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

def heuristic(G: dict, node, coeff: float):
    """计算节点优先级"""
    nbr_ids = G[node]
    n = len(nbr_ids)
    if n < 2:
        return len(G[node])
    max_possible = n * (n - 1) / 2
    actual = 0
    for i in range(n):
        u = nbr_ids[i]
        for j in range(i+1, n):
            v = nbr_ids[j]
            if any(nbr == v for nbr in G.get(u, [])):
                actual += 1
    clustering = actual / max_possible if max_possible > 0 else 0
    return len(G[node]) + coeff * clustering

def get_sorted_list_by_heuristic(G: dict, coeff: float):
    """按启发式值降序排序，相同分数内随机打乱"""
    scores = {node: heuristic(G, node, coeff) for node in G}
    from collections import defaultdict
    groups = defaultdict(list)
    for node, score in scores.items():
        groups[score].append((node, score))
    
    randomized_items = []
    for score in sorted(groups.keys(), reverse=True):
        group = groups[score]
        random.shuffle(group)
        randomized_items.extend(group)
    return randomized_items

def remove_node_from_dict(G: dict, node):
    """删除节点及其所有边"""
    for nxt in G[node]:
        G[nxt].remove(node)
    del G[node]

def iterate_remove_node(G, target, coeff, update_freq=5):
    """
    迭代删除节点，返回每次删除后的最大连通分量大小列表
    G: 原始图（会被修改）
    target: 停止条件（最大连通分量大小 <= target）
    coeff: 启发式系数
    """
    res = []
    while True:
        seq = get_sorted_list_by_heuristic(G, coeff)
        for i in range(update_freq):
            del_node = seq[i][0]
            remove_node_from_dict(G, del_node)
            cv = max_link(G)
            res.append(cv)
            if cv <= target:
                return res
    return res

def main():
    # 尝试的系数列表
    coeff_list = [0.0]
    print(coeff_list)
    # 存储结果的DataFrame：行=系数，列=城市
    results_df = pd.DataFrame(index=coeff_list, columns=[os.path.basename(f).replace("_filtered_Edgelist.csv", "") for f in city_files])
    
    for coeff in coeff_list:
        print(f"\n=== 正在处理系数: {coeff} ===")
        for filepath in city_files:
            city_name = os.path.basename(filepath).replace("_filtered_Edgelist.csv", "")
            print(f"  城市: {city_name}")
            
            # 读取数据并构建图
            df = pd.read_csv(filepath)
            G = build_graph(df)
            node_cnt = len(G)
            target = node_cnt * 0.01
            
            res = iterate_remove_node(G, target, coeff, update_freq=50)

            # 绘图：移除节点比例 vs 最大连通分量大小
            x = [i / node_cnt for i in range(len(res) + 1)]
            y = [1.0] + [r / node_cnt for r in res]
            plt.figure()  # 新窗口绘制最终曲线
            plt.plot(x, y, marker='o', linestyle='-', linewidth=1.5, markersize=3)
            plt.xlabel("Fraction of Removed Nodes")
            plt.ylabel("Largest Component Size")
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 0.5)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.title(f"{city_name}")
            plt.savefig(f"figures/{city_name}_with_degree_heuristic.png")
            
            area = sum(res) / (node_cnt * node_cnt)
            results_df.loc[coeff, city_name] = area
            print(f"    面积 = {area:.6f}")
    
    # 保存结果到CSV
    results_df.to_csv("heuristic_coefficient_results.csv", float_format="%.6f")
    print("\n结果已保存到 heuristic_coefficient_results.csv")
    print(results_df)

if __name__ == "__main__":
    main()