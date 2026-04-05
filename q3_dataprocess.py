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





# 使用示例
if __name__ == "__main__":
    city_names = ["Chengdu","Dalian","Dongguan","Harbin","Qingdao","Quanzhou","Shenyang","Zhengzhou"]
    
    data = [
            [0.049454 , 0.042808 , 0.044988 , 0.047182 , 0.047230 , 0.049241 , 0.049410 , 0.059691 , 0.065801 , 0.083577],
            [0.031542 , 0.027377 , 0.028852 , 0.029564 , 0.029191 , 0.030963 , 0.034448 , 0.037988 , 0.035220 , 0.048947],
            [0.045443 , 0.042354 , 0.039997 , 0.041200 , 0.042377 , 0.042744 , 0.045962 , 0.053730 , 0.057238 , 0.075980],
            [0.045807 , 0.039810 , 0.043272 , 0.048433 , 0.051488 , 0.053339 , 0.053317 , 0.056905 , 0.064578 , 0.082521],
            [0.036208 , 0.031444 , 0.030679 , 0.031273 , 0.030963 , 0.033733 , 0.035342 , 0.035708 , 0.041509 , 0.044791],
            [0.067718 , 0.054743 , 0.054149 , 0.053596 , 0.053547 , 0.055953 , 0.061102 , 0.066473 , 0.065843 , 0.083457],
            [0.057577 , 0.052072 , 0.048764 , 0.054185 , 0.059513 , 0.059557 , 0.061525 , 0.070516 , 0.081701 , 0.104561],
            [0.062269 , 0.052817 , 0.056697 , 0.057674 , 0.056898 , 0.061811 , 0.068276 , 0.079858 , 0.084850 , 0.109259],
            ]

    for i in range(len(city_names)):

        img_path = Path(__file__).parent / "r_decisive" / f"{city_names[i]}_r_decisive.png"
        city_name = city_names[i]





        y = data[i]
        x = [0,50,100,200,300,400,500,750,1000,2000]

    # 绘图
        plt.figure(figsize=(10,6))
        plt.plot(x, y, marker='.', linestyle='-')
        plt.title(f"r_decisive - {city_name}")
        plt.xlabel("Radius of Breaking Area")
        plt.ylabel("Minimized Robustness")
        plt.grid(True)
        plt.ylim(0,max(y)+0.02)
        plt.xlim(0,2050)
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.show()
        print(f"图片保存至: {img_path}")


