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
const int N=1e5,M=2e5,inf=1e9;

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

struct eg{
    int d,id;
};
struct eg_tag{
    int a,b;
    double dis;
    bool operator < (const eg_tag& other) const{
        return other.dis<dis;
    };
};
bool deleted[N],in_max_comp[N];
vector<eg> g[N],g1[N];
int s,t,cnt=0,v[M],v1[M],pos[N],dis[N],node_cnt;
bool found[M],vis[N];
int node_idx[M],rnode_idx[M];
vector<int> max_comp;
eg flow[N];
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
    for(int i=1;i<N;i++) g1[i].clear();
    memset(v1,0,sizeof(v1));
    memset(found,0,sizeof(found));
    ifstream file(filepath);
    string line;
    getline(file,line);
    node_cnt=1;
    cnt=0;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<string> row;
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        int a = stoi(row[2]);
        int b = stoi(row[3]);
        if(!found[a]){
            node_idx[a]=++node_cnt;
            rnode_idx[node_cnt]=a;
            found[a]=true;
        }
        a=node_idx[a];
        if(!found[b]){
            node_idx[b]=++node_cnt;
            rnode_idx[node_cnt]=b;
            found[b]=true;
        }
        b=node_idx[b];
        add(a*2+1, b*2, 1);
    }
    for(int i=1;i<=node_cnt;i++) add(i*2,i*2+1,inf);
}

void dinic(int iters){
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

void kruskal(int center){
}

void add_edge(int sum,int update_freq){
    int iters=50;
    while(true){
        dinic(iters);
        vector<eg> candidates;
        for (int i = 1; i <= node_cnt; ++i) {
            candidates.push_back(flow[i]);
        }
        int m = min(update_freq, (int)candidates.size());
        partial_sort(candidates.begin(), candidates.begin()+m, candidates.end(),
                    [](const eg& a, const eg& b) { return a.d > b.d; });
        for(int i=0;i<iters;i++){
            int node=candidates[i].id;
        }
    }
}

// 迭代删除节点
vector<int> iter_delete_nodes(int target, int update_freq) {
    vector<int> ans;
    int iters=50;
    int comp=node_cnt;
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
            if (!deleted[i]) candidates.push_back(flow[i]);
        }
        // 使用 partial_sort 取出前 update_freq 个
        int m = min(update_freq, (int)candidates.size());
        partial_sort(candidates.begin(), candidates.begin()+m, candidates.end(),
                    [](const eg& a, const eg& b) { return a.d > b.d; });
        // 依次删除这些节点
        for (int i = 0; i < iters_; ++i) {
            int node = candidates[i].id;
            deleted[node] = true;
            comp = largest_component_w();
            cout<<comp<<' ';
            ans.push_back(comp);
            if(comp<=target) return ans;
        }
    }
}

int main(){
    for(string filepath:city_files){
        read_csv(filepath);
        int target=node_cnt/100;
        memset(deleted, 0, sizeof(deleted));
        vector<int> res=iter_delete_nodes(target, 50);
        cout << filepath << endl;
        double ans=0;
        for(int x:res) ans+=(double)x/(node_cnt*node_cnt);
        cout<<res.size()<<' ';
        cout<<"result: "<<ans<<endl;
    }
    return 0; 
}