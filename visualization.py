import pygame
import osmnx as ox
import networkx as nx
import numpy as np
import heapq
from shapely.geometry import LineString
from geopy.distance import geodesic
from complexity import algorithm_complexities
pygame.init()

def heuristic_fn(node_a, node_b, G):
    """
    Heuristic function estimating distance (meters) between two nodes in an OSMnx graph
    using geodesic distance.

    Parameters:
        node_a (int): ID of the first node
        node_b (int): ID of the second node
        G (networkx.MultiDiGraph): OSMnx road network graph

    Returns:
        float: Estimated distance in meters
    """
    lat1, lon1 = G.nodes[node_a]['y'], G.nodes[node_a]['x']
    lat2, lon2 = G.nodes[node_b]['y'], G.nodes[node_b]['x']
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def trace_greedy_bfs(start, goal, G):
    """
    Traces Greedy Best-First Search on a graph as a generator.
    Yields (visited, predecessors) at each step.

    Parameters:
        start (int): Starting node ID
        goal (int): Goal node ID
        G (networkx.Graph or adjacency dict): Graph or adjacency dict
    """
    visited = {start: 0}  # visited dict with distance/order info
    predecessors = {start: None}
    open_heap = []

    # Initial heuristic for the start node
    heapq.heappush(open_heap, (heuristic_fn(start, goal, G), start))

    while open_heap:
        h_cur, current = heapq.heappop(open_heap)

        # Skip if already visited
        if current in visited and visited[current] != 0:
            continue

        # Mark current node as visited (distance/order info can be arbitrary for BFS style)
        visited[current] = visited.get(current, 0)

        # Yield current state
        yield visited, predecessors

        # Stop if goal reached
        if current == goal:
            return

        # Get neighbors
        neighbors = G[current] if hasattr(G, 'neighbors') else G.get(current, [])

        for neighbor in neighbors:
            if neighbor not in visited:
                predecessors[neighbor] = current
                h_neighbor = heuristic_fn(neighbor, goal, G)
                heapq.heappush(open_heap, (h_neighbor, neighbor))



def manhattan(a, b):
    """
    Manhattan distance heuristic (for grid-style graphs)

    Parameters:
        a, b: objects with x and y attributes

    Returns:
        float: Manhattan distance
    """
    return abs(a.x - b.x) + abs(a.y - b.y)

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
def draw_complexity_info(screen, algorithm_name):
    if algorithm_name not in algorithm_complexities:
        return

    # Fonts
    heading_font = pygame.font.SysFont("consolas", 22, bold=True)
    body_font = pygame.font.SysFont("consolas", 20)

    # Text surfaces (white for better readability)
    algo_text = heading_font.render(f"Algorithm: {algorithm_name}", True, (255, 255, 255))
    time_text = body_font.render(f"Time Complexity: {algorithm_complexities[algorithm_name]['time']}", True, (255, 255, 255))
    space_text = body_font.render(f"Space Complexity: {algorithm_complexities[algorithm_name]['space']}", True, (255, 255, 255))

    texts = [algo_text, time_text, space_text]

    # Auto box width and height (fit to content + padding)
    padding_x, padding_y = 15, 12
    width = max(t.get_width() for t in texts) + 2 * padding_x
    height = sum(t.get_height() for t in texts) + (len(texts) - 1) * 6 + 2 * padding_y  # added line spacing

    # Box surface with border radius and darker purple bg
    box = pygame.Surface((width, height), pygame.SRCALPHA)
    box.fill((0, 0, 0, 0))  # transparent base

    bg_color = (111, 66, 193, 200)  # darker vibrant purple with alpha
    border_color = (200, 160, 255)  # lighter purple border

    pygame.draw.rect(box, bg_color, box.get_rect(), border_radius=8)
    pygame.draw.rect(box, border_color, box.get_rect(), 2, border_radius=8)

    # Blit text with spacing inside the box
    y = padding_y
    for t in texts:
        box.blit(t, (padding_x, y))
        y += t.get_height() + 6  # vertical spacing

    # Place bottom-left corner
    screen.blit(box, (20, screen.get_height() - height - 20))                
                



