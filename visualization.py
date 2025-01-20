import pygame
import osmnx as ox
import networkx as nx
import numpy as np
import heapq
from shapely.geometry import LineString

# Tracing Dijkstra's Algorithm
def trace_dijkstra(G, start, end):
    queue = [start]
    visited = {start: 0}
    predecessors = {start: None}
    while queue:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            for key in G[current][neighbor]:
                edge_weight = G.edges[current, neighbor, key].get('weight', 1)
                if neighbor not in visited or visited[current] + edge_weight < visited[neighbor]:
                    visited[neighbor] = visited[current] + edge_weight
                    predecessors[neighbor] = current
                    queue.append(neighbor)
                    yield visited, predecessors

# Tracing A-star Algorithm
def trace_a_star(G, start, end):
    def heuristic(u, v):
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    
    open_set = [(0, start)]
    heapq.heapify(open_set)
    visited = {start: 0}
    predecessors = {start: None}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            break
        for neighbor in G.neighbors(current):
            for key in G[current][neighbor]:
                edge_weight = G.edges[current, neighbor, key].get('weight', 1)
                tentative_g_score = visited[current] + edge_weight
                if neighbor not in visited or tentative_g_score < visited[neighbor]:
                    visited[neighbor] = tentative_g_score
                    priority = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (priority, neighbor))
                    predecessors[neighbor] = current
                    yield visited, predecessors

# Tracing Bellman Ford Algorithm
def trace_bellman_ford(G, start, end):
    # Initialize distances and predecessors
    distance = {node: float('inf') for node in G.nodes}
    distance[start] = 0
    predecessors = {node: None for node in G.nodes}

    # Initialize visited dictionary for consistency with other algorithms
    visited = {}

    # Relax edges |V| - 1 times
    for _ in range(len(G.nodes) - 1):
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessors[v] = u
                visited[v] = True  # Mark node as visited
        
        yield visited, predecessors

    # Check for negative weight cycles
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        if distance[u] + weight < distance[v]:
            print("Graph contains a negative weight cycle")
            return

    yield visited, predecessors


# Tracing BFS Algorithm
def trace_bfs(G, start, end):
    queue = [start]
    visited = {start: 0}
    predecessors = {start: None}
    while queue:
        current = queue.pop(0)
        if current == end:
            break
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1
                predecessors[neighbor] = current
                queue.append(neighbor)
                yield visited, predecessors


# Tracing DFS Algorithm
def trace_dfs(G, start, end):
    stack = [start]
    visited = {start: 0}
    predecessors = {start: None}
    while stack:
        current = stack.pop()
        if current == end:
            break
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1
                predecessors[neighbor] = current
                stack.append(neighbor)
                yield visited, predecessors

