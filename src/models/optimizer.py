"""
A* Route Optimizer
Uses GNN-predicted edge costs to find optimal maritime routes between ports.
"""

import heapq
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

from src.models.gnn import MaritimeGNN, build_pyg_graph, load_model, DEVICE
from src.features.engineer import haversine_km

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of a route optimization query."""
    origin: str
    destination: str
    path_ports: list
    path_coords: list
    total_cost: float
    total_distance_km: float
    n_hops: int
    found: bool = True


# ── 1. Heuristic ──────────────────────────────────────────────────────────────

def haversine_heuristic(port_idx: int, goal_idx: int, nodes_df: pd.DataFrame) -> float:
    """
    A* admissible heuristic: great-circle distance to goal (normalized).
    Never overestimates true cost → guarantees optimal path.
    """
    src = nodes_df.iloc[port_idx]
    dst = nodes_df.iloc[goal_idx]
    dist = haversine_km(src["lat"], src["lon"], dst["lat"], dst["lon"])
    # Normalize by max possible distance (~20,000 km) and scale to cost range
    return (dist / 20000.0) * 0.1


# ── 2. A* Algorithm ───────────────────────────────────────────────────────────

def astar(
    start_idx: int,
    goal_idx: int,
    edge_index: np.ndarray,
    edge_costs: np.ndarray,
    nodes_df: pd.DataFrame,
) -> tuple[list, float]:
    """
    A* pathfinding on the port graph.

    Args:
        start_idx  : Index of departure port in nodes_df
        goal_idx   : Index of arrival port in nodes_df
        edge_index : [2, E] array of (src, dst) edges
        edge_costs : [E] array of GNN-predicted costs
        nodes_df   : Port DataFrame with lat/lon

    Returns:
        (path, total_cost) where path is list of port indices
    """
    # Build adjacency list: {src: [(dst, cost, edge_idx), ...]}
    adjacency = {}
    for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append((dst, float(edge_costs[i])))

    # Priority queue: (f_score, g_score, port_idx, path)
    h0 = haversine_heuristic(start_idx, goal_idx, nodes_df)
    heap = [(h0, 0.0, start_idx, [start_idx])]
    visited = {}

    while heap:
        f, g, current, path = heapq.heappop(heap)

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        # Goal reached
        if current == goal_idx:
            return path, g

        # Explore neighbors
        for neighbor, cost in adjacency.get(current, []):
            new_g = g + cost
            h = haversine_heuristic(neighbor, goal_idx, nodes_df)
            new_f = new_g + h
            heapq.heappush(heap, (new_f, new_g, neighbor, path + [neighbor]))

    return [], float("inf")  # No path found


# ── 3. Route Optimizer ────────────────────────────────────────────────────────

class MaritimeRouteOptimizer:
    """
    End-to-end maritime route optimizer.
    Combines GNN cost prediction with A* pathfinding.
    """

    def __init__(
        self,
        nodes_path: str = "data/processed/graph_nodes.parquet",
        edges_path: str = "data/processed/graph_edges.parquet",
        model_path: str = "data/processed/gnn_model.pt",
    ):
        log.info("Initializing Maritime Route Optimizer...")

        # Load graph
        self.data = build_pyg_graph(nodes_path, edges_path)
        self.nodes_df = self.data.nodes_df.reset_index(drop=True)
        self.edges_df = self.data.edges_df

        # Load GNN and predict costs
        self.model = load_model(
            model_path,
            node_features=self.data.x.shape[1],
            edge_features=self.data.edge_attr.shape[1],
        ).to(DEVICE)

        self.edge_costs = self._predict_costs()
        self.edge_index = self.data.edge_index.numpy()

        log.info(f"  Ports available : {len(self.nodes_df)}")
        log.info(f"  Routes available: {len(self.edges_df)}")
        log.info("✅ Optimizer ready!")

    def _predict_costs(self) -> np.ndarray:
        """Run GNN inference to get edge costs."""
        self.model.eval()
        with torch.no_grad():
            costs = self.model(
                self.data.x.to(DEVICE),
                self.data.edge_index.to(DEVICE),
                self.data.edge_attr.to(DEVICE),
            ).squeeze().cpu().numpy()
        return costs

    def find_port(self, query: str) -> Optional[int]:
        """
        Find port index by name (case-insensitive partial match).
        Returns node index or None.
        """
        query_lower = query.lower().strip()
        matches = self.nodes_df[
            self.nodes_df["name"].str.lower().str.contains(query_lower, na=False)
        ]
        if len(matches) == 0:
            return None
        return matches.index[0]  # Return first match

    def list_ports(self) -> pd.DataFrame:
        """Return all available ports."""
        return self.nodes_df[["name", "country", "lat", "lon"]].copy()

    def optimize(self, origin: str, destination: str) -> RouteResult:
        """
        Find optimal route between two ports.

        Args:
            origin      : Port name (partial match ok, e.g. 'New York')
            destination : Port name (partial match ok, e.g. 'Miami')

        Returns:
            RouteResult with path, cost, distance
        """
        # Find port indices
        src_idx = self.find_port(origin)
        dst_idx = self.find_port(destination)

        if src_idx is None:
            log.warning(f"Port not found: '{origin}'")
            return RouteResult(origin, destination, [], [], 0, 0, 0, found=False)

        if dst_idx is None:
            log.warning(f"Port not found: '{destination}'")
            return RouteResult(origin, destination, [], [], 0, 0, 0, found=False)

        src_port = self.nodes_df.iloc[src_idx]["name"]
        dst_port = self.nodes_df.iloc[dst_idx]["name"]

        log.info(f"\nOptimizing route: {src_port} → {dst_port}")

        # Run A*
        path_indices, total_cost = astar(
            src_idx, dst_idx,
            self.edge_index,
            self.edge_costs,
            self.nodes_df,
        )

        if not path_indices:
            log.warning("No route found between these ports.")
            return RouteResult(origin, destination, [], [], 0, 0, 0, found=False)

        # Build result
        path_ports = [self.nodes_df.iloc[i]["name"] for i in path_indices]
        path_coords = [
            (self.nodes_df.iloc[i]["lat"], self.nodes_df.iloc[i]["lon"])
            for i in path_indices
        ]

        # Total distance
        total_dist = sum(
            haversine_km(
                path_coords[i][0], path_coords[i][1],
                path_coords[i+1][0], path_coords[i+1][1]
            )
            for i in range(len(path_coords) - 1)
        )

        result = RouteResult(
            origin=src_port,
            destination=dst_port,
            path_ports=path_ports,
            path_coords=path_coords,
            total_cost=round(float(total_cost), 4),
            total_distance_km=round(float(total_dist), 1),
            n_hops=len(path_ports) - 1,
            found=True,
        )

        log.info(f"  Path    : {' → '.join(path_ports)}")
        log.info(f"  Hops    : {result.n_hops}")
        log.info(f"  Cost    : {result.total_cost:.4f}")
        log.info(f"  Distance: {result.total_distance_km:.1f} km")

        return result


# ── 4. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    optimizer = MaritimeRouteOptimizer()

    # Show available ports
    print("\n📍 Available ports:")
    print(optimizer.list_ports().to_string(index=False))

    # Test a route
    print("\n" + "="*50)
    result = optimizer.optimize("Houston", "Tampa")

    if result.found:
        print(f"\n✅ Route found!")
        print(f"   {' → '.join(result.path_ports)}")
        print(f"   Distance : {result.total_distance_km} km")
        print(f"   Cost     : {result.total_cost}")
        print(f"   Hops     : {result.n_hops}")
    else:
        print("❌ No route found — try different port names")
        print("Available ports:", optimizer.list_ports()["name"].tolist())