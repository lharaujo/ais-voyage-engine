import calendar

import polars as pl

from src.config import SILVER_DIR, app, get_logger, image, volume
from src.extract import process_daily_ais, run_unlocode_scraper
from src.transform import stitch_voyages

logger = get_logger("orchestrator")


@app.function(image=image, volumes={"/data": volume}, memory=4096, timeout=3600)
def full_pipeline(year: int, month: int):
    """Orchestrates the full Medallion pipeline."""
    logger.info(f"🚀 Starting Pipeline: {year}-{month:02d}")

    # 1. Setup
    run_unlocode_scraper()

    # 2. Bronze: Parallel Map
    days = range(1, calendar.monthrange(year, month)[1] + 1)
    logger.info(f"Spawning {len(days)} Bronze workers...")
    list(extract_worker.map([(year, month, d) for d in days]))

    # 3. Silver: Stitching
    stitch_voyages(year, month)

    # 4. Gold: Parallel Routing
    logger.info("Enriching Gold layer with maritime distances...")
    silver_file = SILVER_DIR / f"silver_{year}_{month:02d}.parquet"
    df = pl.read_parquet(silver_file)
    logger.info(f"Loaded {len(df)} rows")

    # Logic to pair Departure -> Arrival for routing
    # ... (Add pairing logic here) ...

    volume.commit()
    logger.info("✅ Pipeline Complete.")


@app.function(image=image, volumes={"/data": volume}, memory=4096)
def extract_worker(args):
    process_daily_ais(*args)


if __name__ == "__main__":
    # Local trigger
    full_pipeline.remote(2025, 1)
