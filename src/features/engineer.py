"""
Feature Engineering for Maritime Route Optimization.
Computes trajectory-level and segment-level features from clean AIS data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# --- Haversine distance ---
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Compute great-circle distance between consecutive points.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features for each AIS position point based on
    the segment from the previous point of the same vessel.
    """
    df = df.copy().sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    log.info("Computing segment features...")

    # Shift within each vessel group
    grp = df.groupby("mmsi")

    df["lat_prev"] = grp["lat"].shift(1)
    df["lon_prev"] = grp["lon"].shift(1)
    df["sog_prev"] = grp["sog"].shift(1)
    df["cog_prev"] = grp["cog"].shift(1)
    df["ts_prev"]  = grp["timestamp"].shift(1)

    # Distance (km)
    mask = df["lat_prev"].notna()
    df.loc[mask, "distance_km"] = haversine_km(
        df.loc[mask, "lat_prev"].values,
        df.loc[mask, "lon_prev"].values,
        df.loc[mask, "lat"].values,
        df.loc[mask, "lon"].values,
    )

    # Time delta (seconds)
    df["time_delta_s"] = (
        df["timestamp"] - df["ts_prev"]
    ).dt.total_seconds()

    # Speed change (knots)
    df["speed_change"] = (df["sog"] - df["sog_prev"]).abs()

    # Heading change (degrees, circular)
    df["heading_change"] = (df["cog"] - df["cog_prev"]).abs()
    df["heading_change"] = df["heading_change"].apply(
        lambda x: min(x, 360 - x) if pd.notna(x) else np.nan
    )

    # Drop first point of each vessel (no previous point)
    df = df.dropna(subset=["distance_km"])

    # Filter unrealistic segments (teleportation / gap > 2h)
    df = df[df["time_delta_s"] <= 7200]   # max 2 hours gap
    df = df[df["distance_km"] <= 100]     # max 100km in one step

    log.info(f"  → {len(df):,} segments computed")
    return df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    log.info("Computing temporal features...")
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["month"]        = df["timestamp"].dt.month
    return df


def compute_vessel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vessel-level aggregate features."""
    log.info("Computing vessel-level features...")

    vessel_stats = df.groupby("mmsi").agg(
        avg_speed=("sog", "mean"),
        std_speed=("sog", "std"),
        total_distance_km=("distance_km", "sum"),
        n_points=("sog", "count"),
    ).reset_index()

    df = df.merge(vessel_stats, on="mmsi", how="left")
    return df


def clean_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Select final feature columns and drop intermediates."""
    cols_to_keep = [
        # Identity
        "mmsi", "timestamp", "vessel_name", "vessel_category",
        # Position
        "lat", "lon",
        # Kinematics
        "sog", "cog", "distance_km", "time_delta_s",
        "speed_change", "heading_change",
        # Temporal
        "hour_of_day", "day_of_week", "is_weekend", "month",
        # Vessel static
        "length", "width", "draft",
        # Vessel aggregate
        "avg_speed", "std_speed", "total_distance_km", "n_points",
    ]
    available = [c for c in cols_to_keep if c in df.columns]
    return df[available]


def run_feature_engineering(
    input_path: str = "data/processed/ais_clean_combined.parquet",
    output_path: str = "data/processed/ais_features.parquet"
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    log.info(f"Loading clean data from {input_path}...")
    df = pd.read_parquet(input_path)
    log.info(f"  → {len(df):,} rows, {df['mmsi'].nunique():,} vessels")

    df = compute_segment_features(df)
    df = compute_temporal_features(df)
    df = compute_vessel_features(df)
    df = clean_and_select(df)

    # Save
    df.to_parquet(output_path, index=False)

    log.info(f"\n{'='*50}")
    log.info(f"✅ Feature engineering complete!")
    log.info(f"   Rows     : {len(df):,}")
    log.info(f"   Features : {len(df.columns)}")
    log.info(f"   Vessels  : {df['mmsi'].nunique():,}")
    log.info(f"   Saved to : {output_path}")
    log.info(f"\n   Columns  : {list(df.columns)}")

    return df


if __name__ == "__main__":
    run_feature_engineering()