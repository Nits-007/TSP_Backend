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

    valid_edges = None
    if connections:
        valid_edges = set()
        for u, v in connections:
            valid_edges.add((u, v))
            valid_edges.add((v, u))

    for _ in range(n - 1):
        u = -1
        min_key = float('inf')
        for v in range(n):
            if not in_mst[v] and key[v] < min_key:
                min_key = key[v]
                u = v

        in_mst[u] = True

        for v in range(n):
            if valid_edges and (u, v) not in valid_edges:
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


def is_connected(u, v, valid_edges):
    if valid_edges is None:
        return True
    return (u, v) in valid_edges or (v, u) in valid_edges


def find_path_bfs(start, end, valid_edges, n):
    if is_connected(start, end, valid_edges):
        return [start, end]
    
    from collections import deque
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in range(n):
            if neighbor not in visited and is_connected(node, neighbor, valid_edges):
                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    
    return None  


def euler_touring_approx(n, dist, connections=None):
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

    valid_edges = None
    if connections:
        valid_edges = set()
        for u, v in connections:
            valid_edges.add((u, v))
            valid_edges.add((v, u))

    visited = [False] * n
    tour = []
    
    for v in euler_path:
        if not visited[v]:
            visited[v] = True
            tour.append(v)

    if valid_edges:
        final_tour = [tour[0]]
        
        for i in range(1, len(tour)):
            current = final_tour[-1]
            next_node = tour[i]
            
            if is_connected(current, next_node, valid_edges):
                final_tour.append(next_node)
            else:
                path = find_path_bfs(current, next_node, valid_edges, n)
                if path:
                    final_tour.extend(path[1:])
                else:
                    final_tour.append(next_node)
        
        tour = final_tour

    cost = 0.0
    for i in range(len(tour) - 1):
        cost += dist[tour[i]][tour[i + 1]]
    
    if valid_edges and not is_connected(tour[-1], tour[0], valid_edges):
        closing_path = find_path_bfs(tour[-1], tour[0], valid_edges, n)
        if closing_path and len(closing_path) > 2:
            tour.extend(closing_path[1:-1])
            for i in range(len(tour) - 1):
                if i >= len(tour) - len(closing_path) + 1:
                    cost += dist[tour[i]][tour[i + 1]]
    
    cost += dist[tour[-1]][tour[0]]
    
    return tour, cost


@app.post("/tsp")
def solve_tsp(data: CityData):
    n = data.n
    coords = data.coordinates
    connections = data.connections if data.connections else None

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = math.dist(coords[i], coords[j])

    tour, cost = euler_touring_approx(n, dist, connections)

    return {
        "path": tour,  
        "cost": cost
    }
