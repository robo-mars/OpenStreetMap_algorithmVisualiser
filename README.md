# 🗺️ Real-Time Pathfinding Simulator

A web-based application for visualizing and comparing A* and Dijkstra pathfinding algorithms on real-world OpenStreetMap data.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- 🌍 **Real-world maps** from OpenStreetMap (San Francisco, Dubai, London, Tokyo, etc.)
- ⚡ **Side-by-side comparison** of Dijkstra and A* algorithms
- 📊 **Performance metrics**: nodes visited, execution time, distance
- 🎨 **Animated visualization** of search process
- 🖱️ **Interactive**: click to set start/end points

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Map_pathfinding.git
   cd Map_pathfinding
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install flask osmnx networkx numpy geopandas folium scikit-learn
   ```

### Running the Application

1. **Start the server**
   ```bash
   python web_server.py
   ```

2. **Open your browser**
   ```
   http://localhost:8080
   ```

3. **Use the simulator**
   - Select a city from the dropdown
   - Click "Load Map Data"
   - Click on the map to set start point (green)
   - Click again to set end point (red)
   - Click "Run Simulation" to compare algorithms

## Project Structure

```
Map_pathfinding/
├── web_server.py          # Flask web server (entry point)
├── pathfinding.py         # A* and Dijkstra implementations
├── osm_loader.py          # OpenStreetMap data loader
├── templates/
│   └── web_simulator.html # Web interface
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Supported Cities

| City | Region |
|------|--------|
| San Francisco | California, USA |
| Manhattan | New York, USA |
| London | UK |
| Paris | France |
| Tokyo | Japan |
| Seattle | Washington, USA |
| Chicago | Illinois, USA |
| Dubai | UAE |

## Algorithm Details

### Dijkstra's Algorithm
- Explores nodes uniformly based on cumulative cost
- Guarantees shortest path
- Time complexity: O((V + E) log V)

### A* Algorithm
- Uses Euclidean distance heuristic
- Explores fewer nodes by prioritizing toward goal
- Typically 50-80% more efficient than Dijkstra

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/load-location` | POST | Load city map data |
| `/api/run-simulation` | POST | Run pathfinding algorithms |

## Dependencies

- Flask >= 2.0.0
- OSMnx >= 1.3.0
- NetworkX >= 3.0
- NumPy >= 1.24.0
- GeoPandas >= 0.12.0
- scikit-learn >= 1.0.0

## Troubleshooting

### Port already in use
```bash
lsof -ti :8080 | xargs kill -9
python web_server.py
```

### City not loading
Some cities require point-based queries. Dubai is handled automatically. For other cities, ensure the name matches OpenStreetMap naming conventions.

## License

MIT License

## Author

Mariam Hashmi
