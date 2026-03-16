import logging
import sys
from pathlib import Path

import modal

# --- 1. CLOUD INFRASTRUCTURE ---
app = modal.App("ais-voyage-engine")
volume = modal.Volume.from_name("ais-data-store", create_if_missing=True)

# Optimized image for 4GB RAM workers
image = modal.Image.debian_slim().pip_install(
    "polars",
    "duckdb",
    "requests",
    "zstandard",
    "scipy",
    "numpy",
    "searoute",
    "tqdm",
    "beautifulsoup4",
    "pandas",
)

# --- 2. PATHS & CONSTANTS ---
DATA_PATH = Path("/data")
BRONZE_DIR = DATA_PATH / "bronze"
SILVER_DIR = DATA_PATH / "silver"
GOLD_DIR = DATA_PATH / "gold"
UNLOCODE_PATH = DATA_PATH / "reference/ports.parquet"

# --- 3. LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str):
    """Returns a logger instance for the specified module name."""
    return logging.getLogger(name)
