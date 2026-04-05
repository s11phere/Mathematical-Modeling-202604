#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <limits>

class PointFinder {
public:
    /**
     * 构造函数：自动根据数据范围划分网格
     * @param xs 点的x坐标列表
     * @param ys 点的y坐标列表
     * @param ids 点的编号列表（与xs/ys一一对应）
     * @param targetCellsPerDim 目标每个维度上的网格数量（默认 sqrt(N)）
     */
    PointFinder(const std::vector<double>& xs, 
                const std::vector<double>& ys,
                const std::vector<int>& ids,
                int targetCellsPerDim = -1) 
    {
        N = xs.size();
        
        // 计算数据范围
        minX = *std::min_element(xs.begin(), xs.end());
        maxX = *std::max_element(xs.begin(), xs.end());
        minY = *std::min_element(ys.begin(), ys.end());
        maxY = *std::max_element(ys.begin(), ys.end());
        
        // 自动确定网格数量
        if (targetCellsPerDim <= 0) {
            targetCellsPerDim = static_cast<int>(std::sqrt(N));
            if (targetCellsPerDim < 1) targetCellsPerDim = 1;
        }
        // 避免网格过密或过疏
        targetCellsPerDim = std::max(1, std::min(targetCellsPerDim, 1000));
        
        // 计算每个方向上的网格尺寸
        double rangeX = maxX - minX;
        double rangeY = maxY - minY;
        
        gridSizeX = rangeX / targetCellsPerDim;
        gridSizeY = rangeY / targetCellsPerDim;
        gridNumX = targetCellsPerDim + 1;  // 实际网格数（边界可能多一个）
        gridNumY = targetCellsPerDim + 1;
        
        // 初始化网格（二维vector，存储点索引）
        grid.resize(gridNumX * gridNumY);
        
        // 将每个点插入对应网格
        for (size_t i = 0; i < N; ++i) {
            int gx = getGridIndex(xs[i], minX, gridSizeX);
            int gy = getGridIndex(ys[i], minY, gridSizeY);
            if (gx >= 0 && gx < gridNumX && gy >= 0 && gy < gridNumY) {
                grid[gy * gridNumX + gx].push_back(i);
            } else {
                // 理论上不会越界，但保留边界处理
                std::cerr << "警告：点 " << ids[i] << " 超出网格范围\n";
            }
        }
        
        // 保存点坐标和ID供查询时精确计算距离
        this->xs = xs;
        this->ys = ys;
        this->ids = ids;
    }
    
    /**
     * 查找距离 (qx, qy) 不超过 radius 的所有点的编号
     * @param qx 查询点x坐标
     * @param qy 查询点y坐标
     * @param radius 半径（非负数）
     * @return 满足条件的点的编号列表
     */
    std::vector<int> findPointsInRadius(double qx, double qy, double radius) const {
        std::vector<int> result;
        if (radius < 0) return result;
        
        // 确定查询范围涉及的网格索引范围
        int gx0 = getGridIndex(qx - radius, minX, gridSizeX);
        int gx1 = getGridIndex(qx + radius, minX, gridSizeX);
        int gy0 = getGridIndex(qy - radius, minY, gridSizeY);
        int gy1 = getGridIndex(qy + radius, minY, gridSizeY);
        
        // 裁剪到有效网格范围内
        gx0 = std::max(0, gx0);
        gx1 = std::min(gridNumX - 1, gx1);
        gy0 = std::max(0, gy0);
        gy1 = std::min(gridNumY - 1, gy1);
        
        const double r2 = radius * radius;
        for (int gy = gy0; gy <= gy1; ++gy) {
            for (int gx = gx0; gx <= gx1; ++gx) {
                for (int idx : grid[gy * gridNumX + gx]) {
                    double dx = xs[idx] - qx;
                    double dy = ys[idx] - qy;
                    if (dx*dx + dy*dy <= r2) {
                        result.push_back(ids[idx]);
                    }
                }
            }
        }
        return result;
    }
    
private:
    int N;
    double minX, maxX, minY, maxY;
    double gridSizeX, gridSizeY;
    int gridNumX, gridNumY;
    std::vector<std::vector<int>> grid;   // 一维扁平化存储，每个格子存储点的索引列表
    std::vector<double> xs, ys;
    std::vector<int> ids;
    
    // 根据坐标和网格参数计算网格索引
    int getGridIndex(double coord, double minCoord, double cellSize) const {
        int idx = static_cast<int>(std::floor((coord - minCoord) / cellSize));
        // 边界情况：当坐标等于max时，可能得到 gridNum，需要减1
        if (idx < 0) idx = 0;
        if (idx >= gridNumX || idx >= gridNumY) idx = (idx >= gridNumX ? gridNumX-1 : gridNumY-1);
        return idx;
    }
};

// 示例用法
int main() {
    // 模拟一些点：id从0开始
    std::vector<double> xs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> ys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    PointFinder finder(xs, ys, ids);
    
    // 查询(5,5)半径3.0内的点
    auto points = finder.findPointsInRadius(5.0, 5.0, 3.0);
    std::cout << "Points within radius 3.0 around (5,5): ";
    for (int id : points) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    xs={5};
    ys={5};
    ids={1};
    finder=PointFinder(xs,ys,ids);
    points = finder.findPointsInRadius(5.0, 5.0, 3.0);
    std::cout << "Points within radius 3.0 around (5,5): ";
    for (int id : points) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    return 0;
}