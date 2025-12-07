"""
OpenStreetMap data loader using OSMnx
"""

import osmnx as ox
import networkx as nx
from typing import Tuple, List, Optional


class OSMDataLoader:
    """Load and process OpenStreetMap data"""
    
    def __init__(self):
        """Initialize the loader"""
        self.graph = None
        self.location = None
        
    def load_city(self, city_name: str, network_type: str = 'drive') -> nx.MultiDiGraph:
        """
        Load street network for a city
        
        Args:
            city_name: Name of city (e.g., "San Francisco, California")
            network_type: Type of network ('drive', 'walk', 'bike', 'all')
            
        Returns:
            NetworkX graph
        """
        try:
            print(f"Downloading street network for {city_name}...")
            self.graph = ox.graph_from_place(city_name, network_type=network_type)
            self.location = city_name
            print(f"✓ Loaded {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            return self.graph
        except Exception as e:
            print(f"✗ Error loading {city_name}: {e}")
            return None
    
    def load_bbox(self, north: float, south: float, east: float, west: float, 
                  network_type: str = 'drive') -> nx.MultiDiGraph:
        """
        Load street network within a bounding box
        
        Args:
            north, south, east, west: Bounding box coordinates
            network_type: Type of network
            
        Returns:
            NetworkX graph
        """
        try:
            print(f"Downloading street network for bbox...")
            self.graph = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
            self.location = f"bbox({north}, {south}, {east}, {west})"
            print(f"✓ Loaded {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            return self.graph
        except Exception as e:
            print(f"✗ Error loading bbox: {e}")
            return None
    
    def simplify_graph(self) -> nx.DiGraph:
        """
        Simplify and convert graph to directed graph
        
        Returns:
            Simplified directed graph
        """
        if self.graph is None:
            print("✗ No graph loaded")
            return None
        
        print("Simplifying graph...")
        # Convert to undirected
        G = nx.Graph(self.graph)
        # Get largest connected component
        G = nx.DiGraph(G.subgraph(max(nx.connected_components(G), key=len)))
        
        print(f"✓ Simplified to {len(G.nodes())} nodes")
        return G
    
    def get_nearest_node(self, lat: float, lon: float):
        """
        Get nearest node to given coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Node ID
        """
        if self.graph is None:
            print("✗ No graph loaded")
            return None
        
        return ox.nearest_nodes(self.graph, lon, lat)
    
    def get_graph_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of graph
        
        Returns:
            Tuple of (north, south, east, west)
        """
        if self.graph is None:
            return None
        
        lats = [self.graph.nodes[node]['y'] for node in self.graph.nodes()]
        lons = [self.graph.nodes[node]['x'] for node in self.graph.nodes()]
        
        return (max(lats), min(lats), max(lons), min(lons))
    
    def get_node_coordinates(self, node):
        """Get lat/lon of a node"""
        if self.graph is None:
            return None
        
        return (self.graph.nodes[node]['y'], self.graph.nodes[node]['x'])
    
    def visualize_in_folium(self, start_node=None, end_node=None, path=None, visited=None):
        """
        Create interactive Folium map visualization
        
        Args:
            start_node: Start node
            end_node: End node
            path: Path nodes
            visited: Visited nodes
            
        Returns:
            Folium map
        """
        if self.graph is None:
            return None
        
        import folium
        
        # Get center of graph
        bounds = self.get_graph_bounds()
        center_lat = (bounds[0] + bounds[1]) / 2
        center_lon = (bounds[2] + bounds[3]) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        # Draw all edges
        for u, v, data in self.graph.edges(data=True):
            coords = [(self.graph.nodes[u]['y'], self.graph.nodes[u]['x']),
                      (self.graph.nodes[v]['y'], self.graph.nodes[v]['x'])]
            folium.PolyLine(coords, color='gray', weight=1, opacity=0.5).add_to(m)
        
        # Draw visited nodes
        if visited:
            for node in visited:
                lat, lon = self.get_node_coordinates(node)
                folium.CircleMarker([lat, lon], radius=3, color='yellow', 
                                   fill=True, fillColor='yellow', opacity=0.6).add_to(m)
        
        # Draw path
        if path:
            path_coords = [self.get_node_coordinates(node) for node in path]
            folium.PolyLine(path_coords, color='red', weight=2, opacity=1).add_to(m)
        
        # Draw start and end
        if start_node:
            lat, lon = self.get_node_coordinates(start_node)
            folium.Marker([lat, lon], popup='Start', icon=folium.Icon(color='green')).add_to(m)
        
        if end_node:
            lat, lon = self.get_node_coordinates(end_node)
            folium.Marker([lat, lon], popup='End', icon=folium.Icon(color='red')).add_to(m)
        
        return m
