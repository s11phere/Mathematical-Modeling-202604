import pandas as pd
import matplotlib.pyplot as plt
import glob

# 读取所有生成的 CSV 文件
for file in glob.glob("*_res.csv"):
    w=17894
    df = pd.read_csv(file, header=None, names=["step", "component_size"])
    plt.figure()
    plt.plot(df["step"]/w, df["component_size"]/w, marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.xlabel("Deletion Step")
    plt.ylabel("Largest Component Size")
    plt.title(file.replace("_res.csv", ""))
    plt.grid(True)
    plt.savefig(file.replace(".csv", ".png"), dpi=150)
    plt.close()