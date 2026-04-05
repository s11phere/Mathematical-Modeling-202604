import numpy as np
import matplotlib.pyplot as plt

def read_numbers(filename):
    """读取空格/换行分隔的数字，返回一维列表"""
    with open(filename, 'r') as f:
        data = f.read().strip().split()
    return [float(x) for x in data]

def plot_line(data, title="Data Sequence", xlabel="Index", ylabel="Value", savefig=None):
    """绘制折线图"""
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='.', linestyle='-', linewidth=1, markersize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if savefig:
        plt.savefig(savefig, dpi=150)
    else:
        plt.show()

def plot_histogram(data, bins='auto', title="Histogram", xlabel="Value", ylabel="Frequency", savefig=None):
    """绘制直方图"""
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if savefig:
        plt.savefig(savefig, dpi=150)
    else:
        plt.show()

if __name__ == "__main__":
    # 读取数据
    data = read_numbers("data.txt")
    print(f"共读取 {len(data)} 个数字")
    print(f"统计: min={np.min(data):.3f}, max={np.max(data):.3f}, mean={np.mean(data):.3f}, std={np.std(data):.3f}")

    # 绘制折线图（默认）
    plot_line(data, title="Data from data.txt", savefig="line_plot.png")

    # 可选：绘制直方图（取消注释下一行）
    # plot_histogram(data, title="Histogram of Data", savefig="histogram.png")