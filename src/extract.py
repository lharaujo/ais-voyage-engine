import polars as pl
import numpy as np
from scipy.spatial import cKDTree

def filter_pings_by_port(df: pl.DataFrame, port_coords: np.ndarray, port_codes: list) -> pl.DataFrame:
    """Filters raw AIS pings to those within 5km of a port."""
    # Build spatial tree
    tree = cKDTree(np.deg2rad(port_coords))
    ping_coords = np.deg2rad(df.select(["latitude", "longitude"]).to_numpy())
    
    # Query distances (5km / 6371km Earth radius in radians)
    dist, idx = tree.query(ping_coords, k=1)
    mask = dist <= (5.0 / 6371.0)
    
    return df.filter(mask).with_columns([
        pl.Series("port_locode", np.array(port_codes)[idx[mask]]),
        pl.lit("Ping").alias("event")
    ])
