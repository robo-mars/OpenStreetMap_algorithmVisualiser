"""
Real-time interactive pathfinding web server
Flask + WebSocket for live algorithm visualization
"""
from flask import Flask, render_template, request, jsonify
import json
import threading
import time as time_module
from pathfinding import Dijkstra, AStar
from osm_loader import OSMDataLoader
import osmnx as ox
import heapq
import math


app = Flask(__name__, template_folder='templates', static_folder='static')

# Global state
current_graph = None
current_location = "San Francisco, California"
client_connections = []


class RealTimePathfinder:
    """Run pathfinding algorithms with step-by-step updates"""
    
    def __init__(self, graph):
        self.graph = graph
        self.step_callbacks = []
    
    def register_callback(self, callback):
        """Register callback for step updates"""
        self.step_callbacks.append(callback)
    
    def notify_step(self, algorithm_name, step_data):
        """Notify all listeners of a step"""
        for callback in self.step_callbacks:
            callback(algorithm_name, step_data)
    
    def run_dijkstra_steps(self, start, end):
        """Run Dijkstra with step-by-step updates"""
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[start] = 0
        previous = {node: None for node in self.graph.nodes()}
        pq = [(0, start)]
        visited = set()
        all_visited = []
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            all_visited.append(current)
            
            # Send update
            self.notify_step('dijkstra', {
                'type': 'visiting',
                'node': current,
                'visited_count': len(visited),
                'distance': current_distance
            })
            time_module.sleep(0.01)  # Small delay for visualization
            
            if current == end:
                break
            
            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph[current][neighbor]
                if isinstance(edge_data, dict):
                    weight = edge_data.get('length', 1)
                else:
                    weight = edge_data[0].get('length', 1)
                
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.insert(0, current)
            current = previous[current]
        
        return path, all_visited, distances[end]
    
    def run_astar_steps(self, start, end):
        """Run A* with step-by-step updates"""
        # Cache end node coordinates for heuristic
        end_lat = self.graph.nodes[end]['y']
        end_lon = self.graph.nodes[end]['x']
        
        # Scale factor: 1 degree latitude ≈ 111,000 meters
        METERS_PER_DEGREE = 111000
        
        def heuristic(node):
            # Euclidean distance in meters (admissible)
            lat1 = self.graph.nodes[node]['y']
            lon1 = self.graph.nodes[node]['x']
            dlat_m = (lat1 - end_lat) * METERS_PER_DEGREE
            dlon_m = (lon1 - end_lon) * METERS_PER_DEGREE
            return (dlat_m**2 + dlon_m**2) ** 0.5
        
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes()}
        g_score[start] = 0
        
        open_set = [(heuristic(start), start)]
        visited = set()
        all_visited = []
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # Lazy deletion: skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            all_visited.append(current)
            
            # Send update
            self.notify_step('astar', {
                'type': 'visiting',
                'node': current,
                'visited_count': len(visited),
                'distance': g_score[current]
            })
            time_module.sleep(0.01)
            
            if current == end:
                break
            
            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph[current][neighbor]
                if isinstance(edge_data, dict):
                    weight = edge_data.get('length', 1)
                else:
                    weight = edge_data[0].get('length', 1)
                
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        # Reconstruct path
        path = [end]
        current = end
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        
        return path, all_visited, g_score[end]


@app.route('/')
def index():
    """Serve main page"""
    return render_template('web_simulator.html')


@app.route('/api/load-location', methods=['POST'])
def load_location():
    """Load OSM data for a location"""
    global current_graph, current_location
    
    data = request.json
    location = data.get('location', 'San Francisco, California')
    
    # Special handling for locations that need point-based queries
    point_locations = {
        'dubai': (25.2048, 55.2708, 2000),  # lat, lon, dist in meters
    }
    
    try:
        loader = OSMDataLoader()
        print(f"Loading {location}...")
        
        # Check if this location needs point-based loading
        location_lower = location.lower()
        graph = None
        for city_key, (lat, lon, dist) in point_locations.items():
            if city_key in location_lower:
                print(f"Using point-based query for {location}...")
                graph = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
                loader.graph = graph
                loader.location = location
                break
        
        # Default: use place-based query
        if graph is None:
            graph = loader.load_city(location, network_type='drive')
        
        if graph is None:
            return jsonify({'error': 'Failed to load location'}), 400
        
        current_graph = graph
        current_location = location
        
        # Get bounds
        bounds = loader.get_graph_bounds()
        
        return jsonify({
            'success': True,
            'location': location,
            'nodes': len(graph.nodes()),
            'edges': len(graph.edges()),
            'bounds': {
                'north': bounds[0],
                'south': bounds[1],
                'east': bounds[2],
                'west': bounds[3],
                'center': [
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2
                ]
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    """Run both algorithms simultaneously"""
    global current_graph
    
    if current_graph is None:
        return jsonify({'error': 'No graph loaded'}), 400
    
    data = request.json
    start_lat, start_lon = data.get('start', [37.7749, -122.4194])
    end_lat, end_lon = data.get('end', [37.8044, -122.2712])
    
    # Find nearest nodes
    loader = OSMDataLoader()
    loader.graph = current_graph
    start_node = loader.get_nearest_node(start_lat, start_lon)
    end_node = loader.get_nearest_node(end_lat, end_lon)
    
    if start_node is None or end_node is None:
        return jsonify({'error': 'Could not find nodes'}), 400
    
    # Run algorithms in parallel
    results = {
        'dijkstra': None,
        'astar': None,
        'start_node': start_node,
        'end_node': end_node
    }
    
    def run_dijkstra():
        dijkstra = Dijkstra(current_graph)
        path, visited, dist = dijkstra.find_path(start_node, end_node)
        results['dijkstra'] = {
            'path': path,
            'visited': visited,
            'distance': dist,
            'execution_time': dijkstra.execution_time_ms  # timing from inside algorithm loop
        }
    
    def run_astar():
        astar = AStar(current_graph)
        path, visited, dist = astar.find_path(start_node, end_node)
        results['astar'] = {
            'path': path,
            'visited': visited,
            'distance': dist,
            'execution_time': astar.execution_time_ms  # timing from inside algorithm loop
        }
    
    # Run in parallel threads
    t1 = threading.Thread(target=run_dijkstra)
    t2 = threading.Thread(target=run_astar)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    # Convert node IDs to coordinates
    def nodes_to_coords(nodes):
        return [[current_graph.nodes[n]['y'], current_graph.nodes[n]['x']] for n in nodes]
    
    # Calculate distance in kilometers (OSMnx edge lengths are in meters)
    dijkstra_distance_km = results['dijkstra']['distance'] / 1000 if results['dijkstra']['distance'] != float('inf') else 0
    astar_distance_km = results['astar']['distance'] / 1000 if results['astar']['distance'] != float('inf') else 0
    
    return jsonify({
        'success': True,
        'dijkstra': {
            'path': nodes_to_coords(results['dijkstra']['path']),
            'visited': nodes_to_coords(results['dijkstra']['visited']),
            'visited_count': len(results['dijkstra']['visited']),
            'path_length': len(results['dijkstra']['path']),
            'distance': results['dijkstra']['distance'],
            'distance_km': round(dijkstra_distance_km, 2),
            'execution_time_ms': results['dijkstra']['execution_time']
        },
        'astar': {
            'path': nodes_to_coords(results['astar']['path']),
            'visited': nodes_to_coords(results['astar']['visited']),
            'visited_count': len(results['astar']['visited']),
            'path_length': len(results['astar']['path']),
            'distance': results['astar']['distance'],
            'distance_km': round(astar_distance_km, 2),
            'execution_time_ms': results['astar']['execution_time']
        }
    })


@app.route('/api/get-coordinates', methods=['POST'])
def get_coordinates():
    """Get coordinates for a city"""
    data = request.json
    location = data.get('location', 'San Francisco, California')
    
    # Common cities with their coordinates
    cities = {
        'san francisco': [37.7749, -122.4194],
        'new york': [40.7128, -74.0060],
        'london': [51.5074, -0.1278],
        'paris': [48.8566, 2.3522],
        'tokyo': [35.6762, 139.6503],
        'sydney': [-33.8688, 151.2093],
        'toronto': [43.6532, -79.3832],
        'berlin': [52.5200, 13.4050],
        'dubai': [25.2048, 55.2708],
    }
    
    location_lower = location.lower()
    for city, coords in cities.items():
        if city in location_lower:
            return jsonify({'coordinates': coords})
    
    return jsonify({'coordinates': [37.7749, -122.4194]})  # Default to SF


if __name__ == '__main__':
    print("Starting Pathfinding Simulator Web Server...")
    print("Open your browser to: http://localhost:8080")
    app.run(debug=True, port=8080, use_reloader=False)
