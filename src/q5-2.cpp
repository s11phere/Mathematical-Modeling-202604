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
#include<memory>
using namespace std;

random_device rd;
mt19937 gen(rd());
class PointFinder {
public:
    PointFinder() {}
    PointFinder(const vector<double>& xs, const vector<double>& ys, const vector<int>& ids, int targetCellsPerDim = -1)
    {
        N = xs.size();
        minX = *min_element(xs.begin(), xs.end());
        maxX = *max_element(xs.begin(), xs.end());
        minY = *min_element(ys.begin(), ys.end());
        maxY = *max_element(ys.begin(), ys.end());
        if (targetCellsPerDim <= 0) {
            targetCellsPerDim = static_cast<int>(sqrt(N));
            if (targetCellsPerDim < 1) targetCellsPerDim = 1;
        }
        targetCellsPerDim = max(1, min(targetCellsPerDim, 1000));
        double rangeX = maxX - minX;
        double rangeY = maxY - minY;
        gridSizeX = rangeX / targetCellsPerDim;
        gridSizeY = rangeY / targetCellsPerDim;
        gridNumX = targetCellsPerDim + 1;
        gridNumY = targetCellsPerDim + 1;
        grid.resize(gridNumX * gridNumY);
        for (size_t i = 0; i < N; ++i) {
            int gx = getGridIndex(xs[i], minX, gridSizeX);
            int gy = getGridIndex(ys[i], minY, gridSizeY);
            grid[gy * gridNumX + gx].push_back(i);
        }
        this->xs = xs;
        this->ys = ys;
        this->ids = ids;
    }

    vector<int> findPointsInRadius(double qx, double qy, double radius) const {
        vector<int> result;
        if (radius < 0) return result;
        int gx0 = getGridIndex(qx - radius, minX, gridSizeX);
        int gx1 = getGridIndex(qx + radius, minX, gridSizeX);
        int gy0 = getGridIndex(qy - radius, minY, gridSizeY);
        int gy1 = getGridIndex(qy + radius, minY, gridSizeY);
        gx0 = max(0, gx0);
        gx1 = min(gridNumX - 1, gx1);
        gy0 = max(0, gy0);
        gy1 = min(gridNumY - 1, gy1);
        const double r2 = radius * radius;
        for (int gy = gy0; gy <= gy1; ++gy) {
            for (int gx = gx0; gx <= gx1; ++gx) {
                for (int idx : grid[gy * gridNumX + gx]) {
                    double dx = xs[idx] - qx;
                    double dy = ys[idx] - qy;
                    if (dx * dx + dy * dy <= r2) {
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
    vector<vector<int>> grid;
    vector<double> xs, ys;
    vector<int> ids;

    int getGridIndex(double coord, double minCoord, double cellSize) const {
        int idx = static_cast<int>(floor((coord - minCoord) / cellSize));
        if (idx < 0) idx = 0;
        if (idx >= gridNumX || idx >= gridNumY) idx = (idx >= gridNumX ? gridNumX - 1 : gridNumY - 1);
        return idx;
    }
};

class CityProcessor {
private:
    static const int N = 70000;
    static const int M = 200000;
    static const int inf = 1e9;

    struct eg {
        int d, id;
    };
    struct tag {
        int id;
        double val;
    };

    PointFinder finder;
    bool deleted[N];
    bool in_max_comp[N];
    vector<eg> g[N];
    vector<eg> g1[N];
    vector<int> max_comp;
    int s, t;
    int cnt;
    int v[M];
    int v1[M];
    int pos[N];
    int dis[N];
    int node_cnt;
    bool found[M];
    bool vis[N];
    int node_idx[M];
    int rnode_idx[M];
    double x_coord[N];
    double y_coord[N];
    tag flow[N];
    vector<int> surr_nodes_rad[N];

    inline void add(int a, int b, int val) {
        cnt++;
        g1[a].push_back({ b, cnt << 1 });
        g1[b].push_back({ a, (cnt << 1) | 1 });
        v1[cnt << 1] = val;
    }

    int largest_component_w() {
        memset(vis, 0, sizeof(vis));
        int best = 0;
        for (int i = 1; i <= node_cnt; ++i) {
            if (!deleted[i] && !vis[i]) {
                queue<int> q;
                q.push(i * 2 + 1);
                vis[i] = true;
                int cnt = 0;
                while (!q.empty()) {
                    int cur = q.front();
                    q.pop();
                    ++cnt;
                    for (auto [d, id] : g1[cur]) {
                        if (!vis[d / 2] && !deleted[d / 2]) {
                            vis[d / 2] = 1;
                            q.push(d + 1);
                        }
                    }
                }
                best = max(best, cnt);
            }
        }
        return best;
    }

    int largest_component() {
        memset(vis, 0, sizeof(vis));
        int best = 0;
        for (int i = 1; i <= node_cnt; ++i) {
            if (!deleted[i] && !vis[i]) {
                queue<int> q;
                vector<int> cur_comp;
                q.push(i * 2 + 1);
                cur_comp.push_back(i);
                vis[i] = true;
                int cnt = 0;
                while (!q.empty()) {
                    int cur = q.front();
                    q.pop();
                    ++cnt;
                    for (auto [d, id] : g1[cur]) {
                        if (!vis[d / 2] && !deleted[d / 2]) {
                            vis[d / 2] = 1;
                            cur_comp.push_back(d / 2);
                            q.push(d + 1);
                        }
                    }
                }
                if (cnt > best) {
                    best = cnt;
                    max_comp = cur_comp;
                }
            }
        }
        memset(in_max_comp, 0, sizeof(in_max_comp));
        for (auto x : max_comp) in_max_comp[x] = 1;
        return best;
    }

    bool bfs() {
        memset(dis, 63, sizeof(dis));
        queue<int> q;
        q.push(s);
        dis[s] = pos[s] = 0;
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            if (deleted[cur / 2]) continue;
            for (auto [d, id] : g[cur]) {
                if (deleted[d / 2]) continue;
                if (v[id] && dis[d] > inf) {
                    dis[d] = dis[cur] + 1;
                    pos[d] = 0;
                    if (d == t) return true;
                    q.push(d);
                }
            }
        }
        return false;
    }

    int dfs(int cur, int tot) {
        if (deleted[cur / 2]) return 0;
        if (cur == t) return tot;
        int ans = 0, rem;
        for (int i = pos[cur]; i < (int)g[cur].size() && tot; ++i) {
            pos[cur] = i;
            if (v[g[cur][i].id] && dis[g[cur][i].d] == dis[cur] + 1) {
                rem = dfs(g[cur][i].d, min(v[g[cur][i].id], tot));
                if (!rem) dis[g[cur][i].d] = inf;
                else {
                    ans += rem;
                    tot -= rem;
                    v[g[cur][i].id] -= rem;
                    v[g[cur][i].id ^ 1] += rem;
                }
            }
        }
        return ans;
    }

    void dinic(int iters = 50) {
        memset(flow, 0, sizeof(flow));
        for (int i = 1; i <= node_cnt; ++i) flow[i].id = i;
        uniform_int_distribution<int> dist(1, node_cnt);
        for (int i = 1; i <= iters; ++i) {
            memcpy(v, v1, sizeof(v));
            memset(pos, 0, sizeof(pos));
            for (int j = 0; j < N; ++j) g[j] = g1[j];

            int s_orig = dist(gen);
            while (deleted[s_orig] || !in_max_comp[s_orig]) s_orig = dist(gen);
            int t_orig = dist(gen);
            while (deleted[t_orig] || t_orig == s_orig || !in_max_comp[t_orig]) t_orig = dist(gen);
            s = s_orig * 2 + 1;
            t = t_orig * 2;

            int ans = 0;
            while (bfs()) ans += dfs(s, inf);

            for (int j = 1; j <= node_cnt; ++j) {
                if (deleted[j]) continue;
                for (auto [d, id] : g[j * 2]) {
                    if (d == j * 2 + 1) {
                        flow[j].val += inf - v[id];
                        break;
                    }
                }
            }
        }
    }

public:
    CityProcessor(const string& filepath) {
        cnt = 0;
        node_cnt = 1;
        memset(deleted, 0, sizeof(deleted));
        memset(in_max_comp, 0, sizeof(in_max_comp));
        memset(v, 0, sizeof(v));
        memset(v1, 0, sizeof(v1));
        memset(pos, 0, sizeof(pos));
        memset(dis, 0, sizeof(dis));
        memset(found, 0, sizeof(found));
        memset(vis, 0, sizeof(vis));
        memset(node_idx, 0, sizeof(node_idx));
        memset(rnode_idx, 0, sizeof(rnode_idx));
        memset(x_coord, 0, sizeof(x_coord));
        memset(y_coord, 0, sizeof(y_coord));
        memset(flow, 0, sizeof(flow));
        for (int i = 0; i < N; ++i) {
            g[i].clear();
            g1[i].clear();
            surr_nodes_rad[i].clear();
        }
        max_comp.clear();

        ifstream file(filepath);
        string line;
        getline(file, line);
        while (getline(file, line)) {
            stringstream ss(line);
            string cell;
            vector<string> row;
            while (getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            double x = stod(row[0]);
            double y = stod(row[1]);
            int a = stoi(row[2]);
            int b = stoi(row[3]);
            if (!found[a]) {
                node_idx[a] = ++node_cnt;
                rnode_idx[node_cnt] = a;
                found[a] = true;
            }
            if (!found[b]) {
                node_idx[b] = ++node_cnt;
                rnode_idx[node_cnt] = b;
                found[b] = true;
            }
            x_coord[node_idx[a]] = x;
            y_coord[node_idx[a]] = y;
            add(node_idx[a] * 2 + 1, node_idx[b] * 2, 1);
        }
        for (int i = 1; i <= node_cnt; ++i) {
            add(i * 2, i * 2 + 1, inf);
        }

        vector<double> X, Y;
        vector<int> idx;
        X.reserve(node_cnt);
        Y.reserve(node_cnt);
        idx.reserve(node_cnt);
        for (int i = 1; i <= node_cnt; ++i) {
            X.push_back(x_coord[i]);
            Y.push_back(y_coord[i]);
            idx.push_back(i);
        }
        finder = PointFinder(X, Y, idx);
    }

    struct eg_tag {
        int a, b;
        double dis;
        bool operator < (const eg_tag& other) const {
            return dis > other.dis;
        };
    };

    inline double get_dis(int a, int b) {
        return sqrt((x_coord[a] - x_coord[b]) * (x_coord[a] - x_coord[b]) + (y_coord[a] - y_coord[b]) * (y_coord[a] - y_coord[b]));
    }

    vector<eg_tag> add_edge(int sum, int update_freq, int rad) {
        memset(deleted, 0, sizeof(deleted));
        int iters = 50;
        int comp = node_cnt;
        int pre_comp = comp;
        int batch = sum / 6;
        uniform_int_distribution<int> dist(1, node_cnt);
        vector<eg_tag> added_edges;
        while (true) {
            comp = largest_component();
            int pre_comp = comp;
            dinic(iters);
            vector<tag> candidates;
            for (int i = 1; i <= node_cnt; ++i) {
                if (!deleted[i]) {
                    candidates.push_back(flow[i]);
                    vector<int> surr_nodes = finder.findPointsInRadius(x_coord[i], y_coord[i], rad);
                    surr_nodes_rad[i].clear();
                    double tot = 0;
                    int tot_cnt = 0;
                    for (int surr : surr_nodes) {
                        if (!deleted[surr]) {
                            surr_nodes_rad[i].push_back(surr);
                            tot_cnt++;
                            tot += flow[surr].val;
                        }
                    }
                    candidates.back().val = tot / tot_cnt;
                }
            }
            int m = min(update_freq, (int)candidates.size());
            partial_sort(candidates.begin(), candidates.begin() + m, candidates.end(),
                [](const tag& a, const tag& b) { return a.val > b.val; });

            int del_cnt = 0;
            for (int i = 0; i < iters; ++i) {
                for (auto cur_del : surr_nodes_rad[candidates[i].id]) {
                    if (!deleted[cur_del]) {
                        deleted[cur_del] = true;
                        del_cnt++;
                        pre_comp = comp;
                        comp = largest_component_w();
                        if (pre_comp - comp > 300 || (double)comp / pre_comp < 0.9) {
                            int fail = 0;
                            for (int j = 0; j < batch; j++) {
                                int a, b, fail_cnt = 0;
                                while (true) {
                                    a = dist(gen), b = dist(gen);
                                    while (deleted[a]) a = dist(gen);
                                    while (deleted[b] || b == a) b = dist(gen);
                                    add(a * 2 + 1, b * 2, 1);
                                    add(b * 2 + 1, a * 2, 1);
                                    if (fail_cnt > 1000) {
                                        fail++;
                                        break;
                                    }
                                    if (get_dis(a, b) > 15000 || largest_component_w() < pre_comp - 100) {
                                        g1[a * 2].pop_back();
                                        g1[a * 2 + 1].pop_back();
                                        g1[b * 2].pop_back();
                                        g1[b * 2 + 1].pop_back();
                                        cnt -= 2;
                                        fail_cnt++;
                                    }
                                    else {
                                        added_edges.push_back({ a,b,get_dis(a,b) });
                                        break;
                                    }
                                }
                                g1[a * 2].pop_back();
                                g1[a * 2 + 1].pop_back();
                                g1[b * 2].pop_back();
                                g1[b * 2 + 1].pop_back();
                                cnt -= 2;
                                comp = largest_component_w();
                            }
                            sum -= batch - fail;
                            if (sum <= 0) return added_edges;
                        }
                        if (del_cnt >= iters) break;
                    }
                }
            }
            if (comp < 1000) break;
        }
        while (sum--) {
            int a = dist(gen), b = dist(gen);
            while (deleted[a]) a = dist(gen);
            while (deleted[b] || b == a) b = dist(gen);
            while (get_dis(a, b) > 15000) {
                a = dist(gen), b = dist(gen);
                while (deleted[a]) a = dist(gen);
                while (deleted[b] || b == a) b = dist(gen);
            }
            added_edges.push_back({ a,b,get_dis(a,b) });
        }
        return added_edges;
    }

    vector<int> run(int target, int update_freq, double rad, int add_sum) {
        vector<eg_tag> added_edges = add_edge(add_sum, update_freq, rad);
        double cost = 0;
        for (auto [a, b, c] : added_edges) {
            add(a * 2 + 1, b * 2, 1);
            add(b * 2 + 1, a * 2, 1);
            cost += c;
        }
        cout << add_sum << " edges added with cost " << cost << ", ";
        memset(deleted, 0, sizeof(deleted));
        vector<int> ans;
        int comp = node_cnt;
        int iters = 50;
        while (true) {
            int iters_;
            if (comp < target * 10) {
                iters_ = iters * comp / (target * 10);
            }
            else {
                iters_ = iters;
            }
            comp = largest_component();
            dinic(iters_);

            vector<tag> candidates;
            for (int i = 1; i <= node_cnt; ++i) {
                if (!deleted[i]) {
                    candidates.push_back(flow[i]);
                    vector<int> surr_nodes = finder.findPointsInRadius(x_coord[i], y_coord[i], rad);
                    surr_nodes_rad[i].clear();
                    double tot = 0;
                    int tot_cnt = 0;
                    for (int surr : surr_nodes) {
                        if (!deleted[surr]) {
                            surr_nodes_rad[i].push_back(surr);
                            tot_cnt++;
                            tot += flow[surr].val;
                        }
                    }
                    candidates.back().val = tot / tot_cnt;
                }
            }
            int m = min(update_freq, (int)candidates.size());
            partial_sort(candidates.begin(), candidates.begin() + m, candidates.end(),
                [](const tag& a, const tag& b) { return a.val > b.val; });

            int del_cnt = 0;
            for (int i = 0; i < iters_; ++i) {
                for (auto cur_del : surr_nodes_rad[candidates[i].id]) {
                    if (!deleted[cur_del]) {
                        deleted[cur_del] = true;
                        del_cnt++;
                        comp = largest_component_w();
                        ans.push_back(comp);
                    }
                    if (del_cnt >= iters_) break;
                    if (comp <= target) return ans;
                }
            }
        }
    }

    int getNodeCount() const { return node_cnt; }
};

int main() {
    // freopen("run_info/q4_info.txt","w",stdout);
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
    int R = 50;
    int add_sum = 100;
    cout << endl << "Radious: " << R << endl;
    for (string filepath : city_files) {
        cout << endl << filepath << endl;
        for (int i = 1; i <= 3; i++) {
            auto processor = make_unique<CityProcessor>(filepath);  // 堆分配
            int target = processor->getNodeCount() / 100;
            vector<int> res = processor->run(target, 50, R, add_sum);
            double ans = 0;
            for (int x : res) ans += (double)x / (processor->getNodeCount() * processor->getNodeCount());
            cout << res.size() << " nodes removed, ";
            cout << "result: " << ans << endl;
        }
    }
    return 0;
}