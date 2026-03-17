import logging
import os
import sys
from pathlib import Path

import modal

# --- 1. CLOUD INFRASTRUCTURE ---
# Define the image FIRST
image = (
    modal.Image.debian_slim(python_version="3.12").pip_install(
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
    # This is key: it tells Modal that 'src' is a package
    .add_local_python_source("src")
)

# Pass the image to the App definition

app = modal.App("ais-voyage-engine")
volume = modal.Volume.from_name("ais-data-store", create_if_missing=True)

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
REFERENCE_DIR = DATA_PATH / "reference"
PORTS_PATH = REFERENCE_DIR / "ports.parquet"

# Ensure local directories exist for developer ergonomics
if not os.environ.get("MODAL_IMAGE_ID"):
    for path in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, REFERENCE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# --- 3. LOGGING CONFIGURATION ---
def get_logger(name: str):
    """Returns a standardized logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --- 4. UTILITIES ---


def call_searoute(origin_coords, dest_coords):
    """
    Calculates maritime distance.
    Lazy-loaded import to keep config.py lightweight for other modules.
    """
    # Fixed F821: Parameters now correctly passed to function body
    try:
        import searoute

        route = searoute.searoute(origin_coords, dest_coords)
        return route["properties"]["length"]
    except Exception:  # Fixed E722: Now catching Exception explicitly
        return None
