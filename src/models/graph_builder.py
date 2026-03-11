"""
Port Graph Builder
Constructs a graph where nodes = ports and edges = historical AIS routes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from dataclasses import dataclass, field
from typing import Optional
from src.features.engineer import haversine_km

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


@dataclass
class Port:
    port_id: int
    name: str
    country: str
    lat: float
    lon: float
    water_body: str = ""


@dataclass
class RouteEdge:
    src_port_id: int
    dst_port_id: int
    distance_km: float
    n_vessels: int = 0
    avg_speed: float = 0.0
    vessel_categories: list = field(default_factory=list)


def load_ports(
    ports_path: str = "data/external/world_ports.csv",
    bbox: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Load world ports and optionally filter by bounding box.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
              e.g. (-100, 20, -60, 50) for US East Coast + Gulf
    """
    df = pd.read_csv(ports_path, encoding="utf-8-sig")

    # Standardize columns
    df = df.rename(columns={
        "Main Port Name": "name",
        "Country Code": "country",
        "Latitude": "lat",
        "Longitude": "lon",
        "World Water Body": "water_body",
        "World Port Index Number": "wpi_number",
    })

    df = df[["wpi_number", "name", "country", "lat", "lon", "water_body"]].dropna(
        subset=["lat", "lon"]
    )
    df["port_id"] = range(len(df))

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        df = df[
            (df["lon"] >= min_lon) & (df["lon"] <= max_lon) &
            (df["lat"] >= min_lat) & (df["lat"] <= max_lat)
        ]
        log.info(f"Filtered to bbox {bbox}: {len(df)} ports")

    log.info(f"Loaded {len(df)} ports")
    return df.reset_index(drop=True)


def assign_port_to_vessel(
    ais_df: pd.DataFrame,
    ports_df: pd.DataFrame,
    radius_km: float = 5.0
) -> pd.DataFrame:
    """
    For each AIS trajectory, detect departure and arrival ports
    by finding the nearest port within radius_km at start and end.
    """
    log.info("Assigning ports to vessel trajectories...")

    port_coords = ports_df[["port_id", "lat", "lon"]].values

    results = []
    vessels = ais_df.groupby("mmsi")

    for mmsi, traj in vessels:
        # First and last position
        first = traj.iloc[0]
        last = traj.iloc[-1]

        # Find nearest port at departure
        dists_dep = haversine_km(
            first["lat"], first["lon"],
            port_coords[:, 1], port_coords[:, 2]
        )
        dep_idx = np.argmin(dists_dep)
        dep_dist = dists_dep[dep_idx]

        # Find nearest port at arrival
        dists_arr = haversine_km(
            last["lat"], last["lon"],
            port_coords[:, 1], port_coords[:, 2]
        )
        arr_idx = np.argmin(dists_arr)
        arr_dist = dists_arr[arr_idx]

        # Only keep if both endpoints are within radius
        if dep_dist <= radius_km and arr_dist <= radius_km:
            dep_port_id = int(port_coords[dep_idx, 0])
            arr_port_id = int(port_coords[arr_idx, 0])

            # Skip trivial routes (same port)
            if dep_port_id == arr_port_id:
                continue

            results.append({
                "mmsi": mmsi,
                "departure_port_id": dep_port_id,
                "arrival_port_id": arr_port_id,
                "departure_port": ports_df.loc[
                    ports_df["port_id"] == dep_port_id, "name"
                ].values[0],
                "arrival_port": ports_df.loc[
                    ports_df["port_id"] == arr_port_id, "name"
                ].values[0],
                "dep_dist_km": round(dep_dist, 2),
                "arr_dist_km": round(arr_dist, 2),
                "vessel_category": traj["vessel_category"].iloc[0],
                "avg_speed": traj["sog"].mean(),
                "n_points": len(traj),
                "route_distance_km": traj["distance_km"].sum(),
                "departure_time": traj["timestamp"].iloc[0],
                "arrival_time": traj["timestamp"].iloc[-1],
            })

    routes_df = pd.DataFrame(results)
    log.info(f"  → {len(routes_df):,} valid routes detected ({len(results)} vessel trips)")
    return routes_df


def build_edge_list(routes_df: pd.DataFrame, min_routes: int = 2) -> pd.DataFrame:
    """
    Aggregate individual vessel routes into graph edges.
    Each edge = (src_port, dst_port) with aggregated statistics.
    """
    log.info("Building edge list from routes...")

    edges = routes_df.groupby(
        ["departure_port_id", "arrival_port_id",
         "departure_port", "arrival_port"]
    ).agg(
        n_vessels=("mmsi", "count"),
        avg_speed=("avg_speed", "mean"),
        avg_distance_km=("route_distance_km", "mean"),
        vessel_categories=("vessel_category", lambda x: list(x.unique())),
    ).reset_index()

    # Filter edges with enough historical data
    edges = edges[edges["n_vessels"] >= min_routes]

    log.info(f"  → {len(edges):,} edges (min {min_routes} vessels each)")
    return edges


def build_graph(
    ais_path: str = "data/processed/ais_features.parquet",
    ports_path: str = "data/external/world_ports.csv",
    output_dir: str = "data/processed",
    radius_km: float = 5.0,
    min_routes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full graph construction pipeline."""

    log.info("=" * 50)
    log.info("Building Maritime Port Graph")
    log.info("=" * 50)

    # Load data
    ais_df = pd.read_parquet(ais_path)
    log.info(f"AIS data: {len(ais_df):,} rows, {ais_df['mmsi'].nunique():,} vessels")

    # Filter ports to AIS coverage area (US waters)
    ports_df = load_ports(
        ports_path,
        bbox=(-170, 10, 150, 55)  # broad coverage for US waters
    )

    # Assign ports to trajectories
    routes_df = assign_port_to_vessel(ais_df, ports_df, radius_km=radius_km)

    if len(routes_df) == 0:
        log.warning("No routes found! Try increasing radius_km.")
        return ports_df, pd.DataFrame(), pd.DataFrame()

    # Build edges
    edges_df = build_edge_list(routes_df, min_routes=min_routes)

    # Keep only ports that appear in edges
    active_port_ids = set(
        edges_df["departure_port_id"].tolist() +
        edges_df["arrival_port_id"].tolist()
    )
    active_ports = ports_df[ports_df["port_id"].isin(active_port_ids)].copy()

    # Save
    out = Path(output_dir)
    routes_df.to_parquet(out / "routes.parquet", index=False)
    edges_df.to_parquet(out / "graph_edges.parquet", index=False)
    active_ports.to_parquet(out / "graph_nodes.parquet", index=False)

    log.info("\n✅ Graph built!")
    log.info(f"   Nodes (active ports) : {len(active_ports)}")
    log.info(f"   Edges (routes)       : {len(edges_df)}")
    log.info(f"   Individual routes    : {len(routes_df)}")
    log.info(f"   Saved to             : {output_dir}/")

    return active_ports, edges_df, routes_df


if __name__ == "__main__":
    build_graph()