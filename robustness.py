import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import random
import matplotlib.pyplot as plt

def read_road_network(csv_file_path):
    """
    读取道路网络CSV文件，返回邻接表字典。
    
    参数:
        csv_file_path (str): CSV文件路径
    
    返回:
        graph_dict (dict): 邻接表，例如 {node: [neighbor1, neighbor2, ...]}
    """
    df = pd.read_csv(csv_file_path)
    
    # 检查必要的列是否存在
    required_columns = ['START_NODE', 'END_NODE']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")
    
    # 使用 set 避免重复边
    graph = defaultdict(set)
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        graph[u].add(v)
        # 如需无向图（计算连通分量时常用），取消下面一行的注释
        # graph[v].add(u)
    
    # 将 set 转换为 list 以便后续使用
    graph_dict = {k: list(v) for k, v in graph.items()}
    return graph_dict


def pop_p(graph_dic,p): # 输入待踢点地图与踢点概率
    n_tot = len(graph_dic)
    n_pop = int(n_tot*p)
    hitler_list = set(random.sample(list(graph_dic.keys()),n_pop)) # 待踢点集合
    
    graph_new = {}
    for nod in graph_dic.keys():
        if nod in hitler_list:
            continue
        else:
            l = []
            for i in graph_dic[nod]:
                if i in hitler_list:
                    continue
                else:
                    l.append(i)
            graph_new[nod] = l
    return(graph_new)



def strenth_line(graph_dic,N): # 效能曲线绘制函数，N为蒙特卡洛次数
    size = len(graph_dic)
    l = []
    d = 150 # 超参数d决定分辨率
    for i in range(0,d+1):
        temp = 0
        print("p=",i/d)
        if i == 0:
            l.append(max_link(graph_dic)/size)
            continue
        elif i == d :
            l.append(0)
            continue
        else:
            for _ in range(N):
                poped_graph = pop_p(graph_dic,i/d)
                temp += max_link(poped_graph)/size
            l.append(temp/N)
    
    # ========== 绘图部分 ==========
    # 横坐标：移除节点的比例（0 到 1）
    x = [i / d for i in range(d + 1)]
    y = l
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=1.5, markersize=3)
    plt.title("Network Robustness Curve")
    plt.xlabel("Fraction of Removed Nodes")
    plt.ylabel("Normalized Largest Component Size")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    return l



# 对于一张给定图判断最大联通分量
def max_link(graph_dic): # 输入为一个存储了节点的字典
    linked_group = [] # 用于存储相互连接的点集
    visited = set()
    max_size = 0
    
    for nod in graph_dic.keys():
        if nod in visited:
            continue
        else: # 深搜扩展
            tep_size = 1
            stack = []
            visited.add(nod)
            for son in graph_dic[nod]:
                if not son in visited:
                    stack.append(son)
                    visited.add(son)
            while stack:
                cur = stack.pop()
                tep_size += 1
                for grandson in graph_dic[cur]:
                    if grandson in visited:
                        continue
                    else:
                        stack.append(grandson)
                        visited.add(grandson)
        if tep_size > max_size:
            max_size = tep_size
    return(max_size)




def main():
    # 请将此处替换为你的CSV文件实际路径
    csv_file = Path(__file__).parent / 'cases' / 'Qingdao_Edgelist.csv'
    
    # 读取数据
    G = read_road_network(csv_file)
    
    # 打印基本信息
    print(f"成功读取道路记录")
    

    print("\n数据读取完成。")
    print(len(G))
    print(max_link(G))

    strenth_line(G,50)

if __name__ == "__main__":
    main()