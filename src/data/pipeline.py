"""
AIS Data Cleaning Pipeline
Transforms raw GeoParquet AIS data into clean, model-ready DataFrames.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# --- Constants ---
VESSEL_TYPES_OF_INTEREST = {
    60: "Passenger",
    61: "Passenger",
    62: "Passenger",
    63: "Passenger",
    64: "Passenger",
    65: "Passenger",
    66: "Passenger",
    67: "Passenger",
    68: "Passenger",
    69: "Passenger",
    70: "Cargo",
    71: "Cargo",
    72: "Cargo",
    73: "Cargo",
    74: "Cargo",
    75: "Cargo",
    76: "Cargo",
    77: "Cargo",
    78: "Cargo",
    79: "Cargo",
    80: "Tanker",
    81: "Tanker",
    82: "Tanker",
    83: "Tanker",
    84: "Tanker",
    85: "Tanker",
    86: "Tanker",
    87: "Tanker",
    88: "Tanker",
    89: "Tanker",
}

MIN_SOG = 0.5       # knots — filter stopped vessels
MAX_SOG = 30.0      # knots — filter outliers
MIN_TRAJ_POINTS = 10  # minimum points per vessel per day


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw(file_path: Path) -> gpd.GeoDataFrame:
    """Load a raw GeoParquet AIS file."""
    log.info(f"Loading {file_path.name}...")
    gdf = gpd.read_parquet(file_path)
    log.info(f"  → {len(gdf):,} rows loaded")
    return gdf


def extract_coordinates(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extract lat/lon from geometry and drop geometry column."""
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["lon"] = gdf.geometry.x
    df["lat"] = gdf.geometry.y
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    rename_map = {
        "base_date_time": "timestamp",
        "mmsi": "mmsi",
        "sog": "sog",
        "cog": "cog",
        "vessel_type": "vessel_type",
        "vessel_name": "vessel_name",
        "length": "length",
        "width": "width",
        "draft": "draft",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def filter_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stopped vessels and speed outliers."""
    before = len(df)
    df = df.dropna(subset=["sog"])
    df = df[(df["sog"] >= MIN_SOG) & (df["sog"] <= MAX_SOG)]
    after = len(df)
    log.info(f"  Speed filter: {before:,} → {after:,} rows ({before - after:,} removed)")
    return df


def filter_vessel_types(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only cargo, tanker and passenger vessels."""
    before = len(df)
    df = df.dropna(subset=["vessel_type"])
    df = df[df["vessel_type"].isin(VESSEL_TYPES_OF_INTEREST.keys())].copy()
    df["vessel_category"] = df["vessel_type"].map(VESSEL_TYPES_OF_INTEREST)
    after = len(df)
    log.info(f"  Vessel type filter: {before:,} → {after:,} rows ({before - after:,} removed)")
    return df


def filter_short_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Remove vessels with too few position reports."""
    before = len(df)
    traj_counts = df.groupby("mmsi").size()
    valid_mmsi = traj_counts[traj_counts >= MIN_TRAJ_POINTS].index
    df = df[df["mmsi"].isin(valid_mmsi)]
    after = len(df)
    n_vessels = df["mmsi"].nunique()
    log.info(f"  Trajectory filter: {before:,} → {after:,} rows | {n_vessels:,} vessels kept")
    return df


def sort_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by vessel and timestamp for correct trajectory order."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)


def clean_file(file_path: Path, output_dir: Path) -> pd.DataFrame:
    """Full cleaning pipeline for a single AIS file."""
    output_path = output_dir / file_path.name.replace(".parquet", "_clean.parquet")

    if output_path.exists():
        log.info(f"Already processed: {output_path.name}")
        return pd.read_parquet(output_path)

    log.info(f"\n{'='*50}")
    log.info(f"Processing: {file_path.name}")

    gdf = load_raw(file_path)
    df = extract_coordinates(gdf)
    df = rename_columns(df)
    df = filter_speed(df)
    df = filter_vessel_types(df)
    df = filter_short_trajectories(df)
    df = sort_trajectories(df)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    log.info(f"  ✅ Saved: {output_path.name} ({len(df):,} rows, {df['mmsi'].nunique():,} vessels)")
    return df


def run_pipeline(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """Run the full cleaning pipeline on all raw AIS files."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    files = sorted(raw_path.glob("ais-*.parquet"))
    if not files:
        log.error(f"No AIS parquet files found in {raw_dir}")
        return

    log.info(f"Found {len(files)} file(s) to process")

    all_dfs = []
    for f in tqdm(files, desc="Processing AIS files"):
        df = clean_file(f, processed_path)
        all_dfs.append(df)

    # Combine all days
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = processed_path / "ais_clean_combined.parquet"
    combined.to_parquet(combined_path, index=False)

    log.info(f"\n{'='*50}")
    log.info(f"✅ Pipeline complete!")
    log.info(f"   Total rows    : {len(combined):,}")
    log.info(f"   Unique vessels: {combined['mmsi'].nunique():,}")
    log.info(f"   Date range    : {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    log.info(f"   Saved to      : {combined_path}")


if __name__ == "__main__":
    run_pipeline()