"""
A* and Dijkstra pathfinding algorithms implementation
"""

import heapq
import math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class PathfindingAlgorithm:
    """Base class for pathfinding algorithms"""
    
    def __init__(self, graph):
        """
        Initialize with a NetworkX graph
        
        Args:
            graph: NetworkX directed graph with 'geometry', 'y', 'x' attributes
        """
        self.graph = graph
        self.visited_nodes = []
        self.path = []
        self.distance = float('inf')
        
    def _get_distance(self, node1, node2) -> float:
        """Calculate Euclidean distance between two nodes"""
        x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
        x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def _haversine_distance(self, node1, node2) -> float:
        """Calculate haversine distance between two nodes (lat/lon)"""
        lat1 = math.radians(self.graph.nodes[node1]['y'])
        lon1 = math.radians(self.graph.nodes[node1]['x'])
        lat2 = math.radians(self.graph.nodes[node2]['y'])
        lon2 = math.radians(self.graph.nodes[node2]['x'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth radius in km
        
        return c * r


class Dijkstra(PathfindingAlgorithm):
    """Dijkstra's shortest path algorithm"""
    
    def find_path(self, start, end) -> Tuple[List, List, float]:
        """
        Find shortest path using Dijkstra's algorithm
        
        Args:
            start: Starting node
            end: End node
            
        Returns:
            Tuple of (path, visited_nodes, total_distance, execution_time_ms)
        """
        import time as time_module
        
        self.visited_nodes = []
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[start] = 0
        previous = {node: None for node in self.graph.nodes()}
        pq = [(0, start)]
        visited = set()
        
        # Start timing immediately before the main loop
        start_time = time_module.perf_counter()
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            self.visited_nodes.append(current)
            
            if current == end:
                break
            
            for neighbor in self.graph.neighbors(current):
                # Get edge weight (distance)
                edge_data = self.graph[current][neighbor]
                if isinstance(edge_data, dict):
                    # Single edge
                    weight = edge_data.get('length', self._get_distance(current, neighbor))
                else:
                    # Multiple edges
                    weight = edge_data[0].get('length', self._get_distance(current, neighbor))
                
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # End timing immediately after the loop
        end_time = time_module.perf_counter()
        self.execution_time_ms = round((end_time - start_time) * 1000, 2)
        
        # Reconstruct path
        self.path = []
        current = end
        while current is not None:
            self.path.insert(0, current)
            current = previous[current]
        
        self.distance = distances[end]
        return self.path, self.visited_nodes, self.distance


class AStar(PathfindingAlgorithm):
    """A* pathfinding algorithm"""
    
    def find_path(self, start, end) -> Tuple[List, List, float]:
        """
        Find shortest path using A* algorithm
        
        Args:
            start: Starting node
            end: End node
            
        Returns:
            Tuple of (path, visited_nodes, total_distance, execution_time_ms)
        """
        import time as time_module
        
        self.visited_nodes = []
        
        # Cache end node coordinates for heuristic
        end_lat = self.graph.nodes[end]['y']
        end_lon = self.graph.nodes[end]['x']
        
        # Scale factor: 1 degree latitude ≈ 111,000 meters
        # We use meters to match edge weights (which are in meters)
        METERS_PER_DEGREE = 111000
        
        def heuristic(node):
            """Euclidean distance heuristic in meters (admissible)"""
            lat1 = self.graph.nodes[node]['y']
            lon1 = self.graph.nodes[node]['x']
            # Convert degree differences to meters
            dlat_m = (lat1 - end_lat) * METERS_PER_DEGREE
            dlon_m = (lon1 - end_lon) * METERS_PER_DEGREE
            # Return Euclidean distance (squared, then sqrt for admissibility)
            # Using actual distance to ensure admissibility
            return (dlat_m**2 + dlon_m**2) ** 0.5
        
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes()}
        g_score[start] = 0
        
        # Simplified open set: just heapq with lazy deletion (like Dijkstra)
        open_set = [(heuristic(start), start)]
        visited = set()
        
        # Start timing immediately before the main loop
        start_time = time_module.perf_counter()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # Lazy deletion: skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            self.visited_nodes.append(current)
            
            if current == end:
                break
            
            for neighbor in self.graph.neighbors(current):
                # Get edge weight
                edge_data = self.graph[current][neighbor]
                if isinstance(edge_data, dict):
                    weight = edge_data.get('length', self._get_distance(current, neighbor))
                else:
                    weight = edge_data[0].get('length', self._get_distance(current, neighbor))
                
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        # End timing immediately after the loop
        end_time = time_module.perf_counter()
        self.execution_time_ms = round((end_time - start_time) * 1000, 2)
        
        # Reconstruct path
        self.path = [end]
        current = end
        while current in came_from:
            current = came_from[current]
            self.path.insert(0, current)
        
        self.distance = g_score[end]
        return self.path, self.visited_nodes, self.distance
