import logging
import os
import sys
from pathlib import Path

import modal

# --- 1. CLOUD INFRASTRUCTURE ---
app = modal.App("ais-voyage-engine")
volume = modal.Volume.from_name("ais-data-store", create_if_missing=True)

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

# --- 2. PATHS & CONSTANTS (Hybrid Logic) ---

# Check if we are running inside a Modal container
if os.environ.get("MODAL_IMAGE_ID"):
    DATA_PATH = Path("/data")
else:
    # Locally, use the 'data' folder in your project directory
    # .parent.parent moves from src/ to the project root
    DATA_PATH = Path(__file__).resolve().parent.parent / "data"

BRONZE_DIR = DATA_PATH / "bronze"
SILVER_DIR = DATA_PATH / "silver"
GOLD_DIR = DATA_PATH / "gold"
# Note: I added a slash between reference and ports.parquet for safety
UNLOCODE_PATH = DATA_PATH / "reference" / "ports.parquet"

# Ensure local directories exist so the code doesn't crash
if not os.environ.get("MODAL_IMAGE_ID"):
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_PATH / "reference").mkdir(parents=True, exist_ok=True)

# --- 3. LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str):
    """Returns a logger instance for the specified module name."""
    return logging.getLogger(name)
