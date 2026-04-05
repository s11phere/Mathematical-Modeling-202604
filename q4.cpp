#include<iostream>
#include<fstream>
#include<sstream>
#include<cstring>
#include<string>
#include<cstdio>
#include<vector>
#include<queue>
#include<stack>
#include<algorithm>
#include<random>
using namespace std;
const int N=7e4,M=2e5,inf=1e9;

std::random_device rd;
std::mt19937 gen(rd());

string city_files[8] = {
    "cases/Chengdu_filtered_Edgelist.csv",
    "cases/Dalian_filtered_Edgelist.csv",
    "cases/Dongguan_filtered_Edgelist.csv",
    "cases/Harbin_filtered_Edgelist.csv",
    "cases/Qingdao_filtered_Edgelist.csv",
    "cases/Quanzhou_filtered_Edgelist.csv",
    "cases/Shenyang_filtered_Edgelist.csv",
    "cases/Zhengzhou_filtered_Edgelist.csv"
};

class PointFinder {
public:
    PointFinder(){}
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
        
        // 初始化网格
        grid.resize(gridNumX * gridNumY);
        
        // 将每个点插入对应网格
        for (size_t i = 0; i < N; ++i) {
            int gx = getGridIndex(xs[i], minX, gridSizeX);
            int gy = getGridIndex(ys[i], minY, gridSizeY);
            grid[gy * gridNumX + gx].push_back(i);

        }
        
        // 保存点坐标和ID供查询时精确计算距离
        this->xs = xs;
        this->ys = ys;
        this->ids = ids;
    }
    
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
PointFinder finder;
struct eg{
    int d,id;
};
bool deleted[N],in_max_comp[N];
vector<eg> g[N],g1[N];
vector<int> max_comp;
int s,t,cnt=0,v[M],v1[M],pos[N],dis[N],node_cnt;
bool found[M],vis[N];
int node_idx[M],rnode_idx[M];
double x_coord[N],y_coord[N];
eg flow[N];
vector<int> surr_nodes_rad[N];
inline void add(int a,int b,int val){
    cnt++;
    g1[a].push_back({b,cnt<<1});
    g1[b].push_back({a,(cnt<<1)|1});
    v1[cnt<<1]=val;
}

int largest_component_w() {
    memset(vis,0,sizeof(vis));
    int best = 0;
    for (int i = 1; i <= node_cnt; ++i) {
        if (!deleted[i] && !vis[i]) {
            queue<int> q;
            q.push(i*2+1);
            vis[i] = true;
            int cnt = 0;
            while (!q.empty()) {
                int cur = q.front();
                q.pop();
                ++cnt;
                for (auto [d, id] : g1[cur]) {
                    if(!vis[d/2]&&!deleted[d/2]){
                        vis[d/2]=1;
                        q.push(d+1);
                    }
                }
            }
            best=max(best,cnt);
        }
    }
    return best;
}

int largest_component() {
    memset(vis,0,sizeof(vis));
    int best = 0;
    for (int i = 1; i <= node_cnt; ++i) {
        if (!deleted[i] && !vis[i]) {
            queue<int> q;
            vector<int> cur_comp;
            q.push(i*2+1);
            cur_comp.push_back(i);
            vis[i] = true;
            int cnt = 0;
            while (!q.empty()) {
                int cur = q.front();
                q.pop();
                ++cnt;
                for (auto [d, id] : g1[cur]) {
                    if(!vis[d/2]&&!deleted[d/2]){
                        vis[d/2]=1;
                        cur_comp.push_back(d/2);
                        q.push(d+1);
                    }
                }
            }
            if(cnt>best){
                best=cnt;
                max_comp=cur_comp;
            }
        }
    }
    memset(in_max_comp,0,sizeof(in_max_comp));
    for(auto x:max_comp) in_max_comp[x]=1;
    return best;
}

bool bfs(){
    memset(dis, 63, sizeof(dis));
    queue<int> q;
    q.push(s);
    dis[s] = pos[s] = 0;
    while(!q.empty()){
        int cur = q.front(); q.pop();
        if (deleted[cur/2]) continue;
        for(auto [d, id] : g[cur]) {
            if (deleted[d/2]) continue;
            if(v[id] && dis[d] > inf){
                dis[d] = dis[cur] + 1;
                pos[d] = 0;
                if(d == t) return true;
                q.push(d);
            }
        }
    }
    return false;
}
int dfs(int cur,int tot){
    if (deleted[cur/2]) return 0;
    if(cur==t) return tot;
    int ans=0,rem;
    for(int i=pos[cur];i<g[cur].size()&&tot;i++){
        pos[cur]=i;
        if(v[g[cur][i].id]&&dis[g[cur][i].d]==dis[cur]+1){
            rem=dfs(g[cur][i].d,min(v[g[cur][i].id],tot));
            if(!rem) dis[g[cur][i].d]=inf;
            else{
                ans+=rem,tot-=rem;
                v[g[cur][i].id]-=rem,v[g[cur][i].id^1]+=rem;
            }
        }
    }
    return ans;
}

void read_csv(string filepath){
    for(int i=0;i<N;i++) g1[i].clear();
    memset(v1,0,sizeof(v1));
    memset(found,0,sizeof(found));
    ifstream file(filepath);
    string line;
    getline(file,line);
    node_cnt=1;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<string> row;
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        double x=stod(row[0]);
        double y=stod(row[1]);
        int a = stoi(row[2]);
        int b = stoi(row[3]);
        if(!found[a]){
            node_idx[a]=++node_cnt;
            rnode_idx[node_cnt]=a;
            found[a]=true;
        }
        if(!found[b]){
            node_idx[b]=++node_cnt;
            rnode_idx[node_cnt]=b;
            found[b]=true;
        }
        x_coord[node_idx[a]]=x;
        y_coord[node_idx[a]]=y;
        add(node_idx[a]*2+1, node_idx[b]*2, 1);
    }
    for(int i=1;i<=node_cnt;i++) add(i*2,i*2+1,inf);
    vector<double> X,Y;
    vector<int> idx;
    X.reserve(node_cnt);
    Y.reserve(node_cnt);
    idx.reserve(node_cnt);
    for(int i=1;i<=node_cnt;i++){
        X.push_back(x_coord[i]);
        Y.push_back(y_coord[i]);
        idx.push_back(i);
    }
    finder=PointFinder(X,Y,idx);
}

void dinic(int iters=50){
    memset(flow, 0, sizeof(flow));
    for(int i=1;i<=node_cnt;i++) flow[i].id = i;
    
    uniform_int_distribution<int> dist(1,node_cnt);
    for(int i=1;i<=iters;i++){
        memcpy(v, v1, sizeof(v));
        memset(pos, 0, sizeof(pos));
        for(int j=0; j<N; j++) g[j] = g1[j];
        
        int s_orig = dist(gen);
        while(deleted[s_orig]||!in_max_comp[s_orig]) s_orig=dist(gen);
        int t_orig = dist(gen);
        while(deleted[t_orig]||t_orig==s_orig||!in_max_comp[t_orig]) t_orig=dist(gen);
        s = s_orig * 2 + 1;
        t = t_orig * 2;
        
        int ans = 0;
        while(bfs()) ans += dfs(s, inf);
        
        // 统计节点流量（只统计未删除节点）
        for(int j=1;j<=node_cnt;j++){
            if (deleted[j]) continue;
            for(auto [d, id] : g[j*2]) {
                if(d == j*2+1){
                    flow[j].d += inf - v[id];
                    break;
                }
            }
        }
    }
}

int heuristic(int idx){
    int deg=0;
    for(auto[d,id]:g1[idx*2+1]) if(!deleted[id/2]) deg++;
    return deg;
}

// 迭代删除节点
vector<int> iter_delete_nodes(int target, int update_freq,double rad) {
    vector<int> ans;
    int comp=node_cnt;
    int iters=50;
    while (true) {
        int iters_;
        if(comp<target*10){
            iters_=iters*comp/(target*10);
        }
        else iters_=iters;
        comp=largest_component();
        dinic(iters_);
        
        // 收集未删除节点并按流量降序排序
        vector<eg> candidates;
        for (int i = 1; i <= node_cnt; ++i) {
            if (!deleted[i]){
                candidates.push_back(flow[i]);
                vector<int> surr_nodes=finder.findPointsInRadius(x_coord[i],y_coord[i],rad);
                int tot=0;
                for(int surr:surr_nodes){
                    tot+=flow[surr].d;
                }
                candidates.back().d=tot;
                surr_nodes_rad[i]=surr_nodes;
            }
        }
        int m = min(update_freq, (int)candidates.size());
        partial_sort(candidates.begin(), candidates.begin()+m, candidates.end(),
                    [](const eg& a, const eg& b) { return a.d > b.d; });
        // 依次删除这些节点
        int del_cnt=0;
        for (int i = 0; i < iters_; ++i) {
            // cout<<"**"<<surr_nodes_rad[candidates[i].id].size()<<"**";
            for(auto cur_del:surr_nodes_rad[candidates[i].id]){
                if(!deleted[cur_del]){
                    deleted[cur_del]=true;
                    del_cnt++;
                    comp = largest_component_w();
                    cout<<comp<<' ';
                    ans.push_back(comp);
                }
                if(del_cnt>=iters_) break;
                if (comp <= target) {
                    return ans;
                }
            }
        }
    }
}

int main(){
    for(string filepath:city_files){
        read_csv(filepath);
        int target=node_cnt/100;
        memset(deleted, 0, sizeof(deleted));
        vector<int> res=iter_delete_nodes(target,50,0);
        cout << "Finished " << filepath << endl;
        double ans=0;
        for(int x:res) ans+=(double)x/(node_cnt*node_cnt);
        cout<<res.size()<<' ';
        cout<<"result: "<<ans<<endl;
    }
    return 0; 
}