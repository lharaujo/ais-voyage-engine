import modal
from pathlib import Path

# Modal Resources
app = modal.App("ais-voyage-engine")
volume = modal.Volume.from_name("ais-data-store", create_if_missing=True)

# 4GB Optimized Container Image
image = (
    modal.Image.debian_slim()
    .pip_install(
        "polars", "duckdb", "requests", "scipy", "numpy", 
        "searoute", "streamlit", "pydeck", "plotly"
    )
)

# Persistent Cloud Paths
DATA_PATH = Path("/data")
BRONZE_DIR = DATA_PATH / "bronze"
SILVER_DIR = DATA_PATH / "silver"
GOLD_DIR = DATA_PATH / "gold"
