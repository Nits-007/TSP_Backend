from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import math
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TSP Approximation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CityData(BaseModel):
    n: int
    coordinates: List[List[float]] 
    connections: List[List[int]]    


def prim_mst(n, dist, connections=None):
    key = [float('inf')] * n
    parent = [-1] * n
    in_mst = [False] * n
    key[0] = 0

    for _ in range(n - 1):
        u = -1
        min_key = float('inf')
        for v in range(n):
            if not in_mst[v] and key[v] < min_key:
                min_key = key[v]
                u = v

        in_mst[u] = True

        for v in range(n):
            if connections and (u, v) not in connections and (v, u) not in connections:
                continue
            if dist[u][v] < key[v] and not in_mst[v]:
                key[v] = dist[u][v]
                parent[v] = u

    mst_edges = []
    for i in range(1, n):
        if parent[i] != -1:
            mst_edges.append((parent[i], i))
    return mst_edges


def euler_dfs(u, adj, path):
    while adj[u]:
        v = adj[u].pop()
        adj[v].remove(u)
        euler_dfs(v, adj, path)
    path.append(u)


def approx_tsp_2(n, dist, connections=None):
    mst_edges = prim_mst(n, dist, connections)
    adj = [[] for _ in range(n)]
    
    for u, v in mst_edges:
        adj[u].append(v)
        adj[v].append(u)
        adj[u].append(v)
        adj[v].append(u)

    euler_path = []
    euler_dfs(0, adj, euler_path)
    euler_path.reverse()

    visited = [False] * n
    tour = []
    for v in euler_path:
        if not visited[v]:
            visited[v] = True
            tour.append(v)

    cost = 0.0
    for i in range(len(tour) - 1):
        cost += dist[tour[i]][tour[i + 1]]
    cost += dist[tour[-1]][tour[0]]
    return tour, cost


@app.post("/tsp")
def solve_tsp(data: CityData):
    n = data.n
    coords = data.coordinates
    connections = set(tuple(c) for c in data.connections) if data.connections else None

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = math.dist(coords[i], coords[j])

    tour, cost = approx_tsp_2(n, dist, connections)

    return {
        "path": tour,  
        "cost": cost
    }