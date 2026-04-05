import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import os


def parse_q3_data(filename="q3_data.txt"):
    """
    读取并解析 q3_data.txt，返回列表，每个元素为：
    (整数列表, 文件名, 结果整数, 结果浮点数)
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取所有行，去除首尾空白，过滤可能的空行
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    # 每两行一组
    for i in range(0, len(lines), 2):
        ints_line = lines[i]          # 整数 + 文件名行
        res_line = lines[i+1]         # 结果行
        
        # 解析第一行：空格分割，最后一个是文件名，前面都是整数
        parts = ints_line.split()
        filename = parts[-1]           # 例如 "cases/beijing_filtered_Edgelist.csv"
        int_list = list(map(int, parts[:-1]))
        
        # 解析第二行：格式 "整数 result: 浮点数"
        res_parts = res_line.split()
        # 假设格式正确：第一个是整数，第二个是 "result:"，第三个是浮点数
        int_val = int(res_parts[0])
        float_val = float(res_parts[2])  # 直接取第三个元素
        
        data.append((int_list, file_path, int_val, float_val))
    
    return data






def get_data(city_name:str): 
    csv_path = Path(__file__).parent / "cases" / f"{city_name}_filtered_edgelist.csv"
    df = pd.read_csv(csv_path)
    required_cols = ['START_NODE', 'END_NODE', 'LENGTH']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV 缺少必需的列: START_NODE, END_NODE, LENGTH")

    G = nx.Graph()

    # 记录每个节点的坐标（取第一次出现）
    node_coords = {}
    for _, row in df.iterrows():
        u = row['START_NODE']
        v = row['END_NODE']
        # 处理坐标（可选）
        if 'XCoord' in df.columns and 'YCoord' in df.columns:
            if u not in node_coords:
                node_coords[u] = (row['XCoord'], row['YCoord'])
            if v not in node_coords:
                node_coords[v] = (row['XCoord'], row['YCoord'])  # 注意：同一行的两个节点坐标可能不同，这里简化处理
        # 添加无向边（标准化方向，避免重复）
        u, v = sorted([u, v])  # 确保 (min, max)
        length = row['LENGTH']
        if G.has_edge(u, v):
            # 如果边已存在（例如双向记录），累加长度
            G[u][v]['length'] += length
        else:
            G.add_edge(u, v, length=length)

    # 添加节点坐标属性
    for node, (x, y) in node_coords.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y

    print(city_name,"获取成功")
    return(G)  



def parse_q4_data(filename="q4_data.txt"):
    """
    读取并解析 q4_data.txt，返回列表，每个元素为：
    (整数列表, 文件名, 结果整数, 结果浮点数)
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取所有行，去除首尾空白，过滤可能的空行
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    # 每三行一组
    for i in range(0, len(lines)):
        
       # 结果行
        res_line = lines[i]

        print(res_line)
        # 解析第一行：空格分割，最后一个是文件名，前面都是整数
        parts = res_line.split()
        robustness = float(parts[-1])            
        int_list = list(map(int, parts[:-5]))
        
        data.append((int_list, robustness))
        
    
    return data



def parse_q5_data(filename="q5_data1.txt"):
    """
    读取并解析 q5_data.txt，返回列表，每个元素为：
    (整数列表, 文件名, 结果整数, 结果浮点数)
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取所有行，去除首尾空白，过滤可能的空行
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    
    for i in range(0, len(lines)):
        
       # 结果行
        res_line = lines[i]

        
        # 解析第一行：空格分割，最后一个是文件名，前面都是整数
        res_line = res_line.split()        
        res_line = list(map(int,res_line))    
        
        data += res_line
        
    
    return data



# 使用示例
if __name__ == "__main__":
    city_names = ["Chengdu","Dalian","Dongguan","Harbin","Qingdao","Quanzhou","Shenyang","Zhengzhou"]
    
    y = parse_q5_data()
    y = [y[i]/y[0] for i in range(len(y))]
    x = [i/17894 for i in range(len(y))]

    plt.plot(x,y)
    plt.show()


