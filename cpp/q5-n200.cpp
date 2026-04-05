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
        return dis > other.dis;
    };
};
double x_coord[N];
double y_coord[N];
bool deleted[N],in_max_comp[N],added[N];
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
        add(node_idx[a]*2+1, node_idx[b]*2, 1);
    }
    for(int i=1;i<=node_cnt;i++) add(i*2,i*2+1,inf);
}

void dinic(int iters){
    memset(flow, 0, sizeof(flow));
    for(int i=1;i<=node_cnt;i++) flow[i].id = i;
    uniform_int_distribution<int> dist(1,node_cnt);
    // int siz=0;
    // for(int i=1;i<=node_cnt;i++) siz+=g1[i].size();
    // cout<<siz<<' ';
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

inline double get_dis(int a,int b){
    return sqrt((x_coord[a]-x_coord[b])*(x_coord[a]-x_coord[b])+(y_coord[a]-y_coord[b])*(y_coord[a]-y_coord[b]));
}

vector<eg_tag> add_edge(int sum,int update_freq){
    memset(deleted, 0, sizeof(deleted));
    int iters=50;
    int comp=node_cnt;
    int pre_comp=comp;
    int batch=sum/6;
    uniform_int_distribution<int> dist(1,node_cnt);
    vector<eg_tag> added_edges;
    while (true) {
        comp=largest_component();
        dinic(iters);
        vector<eg> candidates;
        for (int i = 1; i <= node_cnt; ++i) {
            if (!deleted[i]) candidates.push_back(flow[i]);
        }
        int m = min(update_freq, (int)candidates.size());
        partial_sort(candidates.begin(), candidates.begin()+m, candidates.end(),
                    [](const eg& a, const eg& b) { return a.d > b.d; });
        for (int i = 0; i < iters; ++i) {
            int node = candidates[i].id;
            deleted[node] = true;
            pre_comp=comp;
            comp = largest_component_w();
            if(pre_comp-comp>300||(double)comp/pre_comp<0.85){
                int fail=0;
                for(int j=0;j<batch;j++){
                    int a,b,fail_cnt=0;
                    while(true){
                        a=dist(gen),b=dist(gen);
                        while(deleted[a]) a=dist(gen);
                        while(deleted[b]||b==a) b=dist(gen);
                        add(a*2+1,b*2,1);
                        add(b*2+1,a*2,1);
                        if(fail_cnt>1000){
                            fail++;
                            break;
                        }
                        if(get_dis(a,b)>10000||largest_component_w()<pre_comp-100){
                            g1[a*2].pop_back();
                            g1[a*2+1].pop_back();
                            g1[b*2].pop_back();
                            g1[b*2+1].pop_back();
                            cnt-=2;
                            fail_cnt++;
                        }
                        else{
                            added_edges.push_back({a,b,get_dis(a,b)});
                            break;
                        }
                    }
                    g1[a*2].pop_back();
                    g1[a*2+1].pop_back();
                    g1[b*2].pop_back();
                    g1[b*2+1].pop_back();
                    cnt-=2;
                    comp=largest_component_w();
                }
                sum-=batch-fail;
                if(sum<=0) return added_edges;
            }
        }
        if(comp<1000) break;
    }
    while(sum--){
        int a=dist(gen),b=dist(gen);
        while(deleted[a]) a=dist(gen);
        while(deleted[b]||b==a) b=dist(gen);
        while(get_dis(a,b)>10000){
            a=dist(gen),b=dist(gen);
            while(deleted[a]) a=dist(gen);
            while(deleted[b]||b==a) b=dist(gen);
        }
        added_edges.push_back({a,b,get_dis(a,b)});
    }
    return added_edges;
}

double random_add_edges(int sum){
    uniform_int_distribution<int> dist(1,node_cnt);
    double ans=0;
    for(int i=1;i<=sum;i++){
        int a=dist(gen);
        int b=dist(gen);
        while(b==a||get_dis(a,b)>100) b=dist(gen);
        cout<<a<<' '<<b<<' '<<get_dis(a,b)<<'|';
        add(a*2+1,b*2,1);
        add(b*2+1,a*2,1);
        ans+=get_dis(a,b);
    }
    return ans;
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
            // cout<<comp<<' ';
            ans.push_back(comp);
            if(comp<=target) return ans;
        }
    }
}

int main(){
    freopen("run_info/q5_info_n200.txt","w",stdout);
    int num_added[1]={200};
    cout<<num_added[0]<<endl;
    for(string filepath:city_files){
        cout <<endl<< filepath << endl;
        for(int num:num_added){
            for(int i=1;i<=3;i++){
                read_csv(filepath);
                int num_add_edges=num;
                vector<eg_tag> added_edges=add_edge(num_add_edges,1000);
                double cost=0;
                for(auto[a,b,c]:added_edges){
                    add(a*2+1,b*2,1);
                    add(b*2+1,a*2,1);
                    cost+=c;
                }
                // double cost=random_add_edges(num_add_edges);
                cout<<num_add_edges<<" edges added with cost "<<cost<<", ";
                int target=node_cnt/100;
                memset(deleted, 0, sizeof(deleted));
                vector<int> res=iter_delete_nodes(target, 50);
                double ans=0;
                for(int x:res) ans+=(double)x/(node_cnt*node_cnt);
                cout<<res.size()<<" nodes removed, ";
                cout<<"result: "<<ans<<endl;
            }
        }
    }
    return 0; 
}