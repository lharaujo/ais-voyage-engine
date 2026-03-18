import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import modal
import polars as pl

from src.config import (
    BRONZE_DIR,
    GOLD_DIR,
    PORTS_PATH,
    REFERENCE_DIR,
    SILVER_DIR,
    app,
    get_logger,
    image,
    volume,
)
from src.extract import AISProcessor, run_unlocode_bootstrap
from src.transform import stitch_voyages

logger = get_logger("orchestrator")

# --- WORKERS & UTILS ---


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("github-auth")],
)
def sync_to_github():
    token = os.environ.get("GH_TOKEN")
    if not token:
        logger.error("GH_TOKEN secret not found.")
        return

    repo_name = "ais-voyage-engine"
    user = "lharaujo"
    repo_url = f"https://{token}@github.com/{user}/{repo_name}.git"
    local_repo = Path("/tmp/ais_repo")
    gold_dir = Path("/data/gold")

    files = sorted(gold_dir.glob("voyages_*.parquet"))
    if not files:
        logger.error("No gold files found in volume to sync!")
        return

    latest_file = files[-1]
    target_file = local_repo / "data" / "gold" / "voyages.parquet"

    try:
        if local_repo.exists():
            shutil.rmtree(local_repo)

        logger.info(f"Cloning {repo_name}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(local_repo)], check=True)

        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(latest_file, target_file)

        os.chdir(local_repo)
        subprocess.run(["git", "config", "user.email", "bot@modal.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Modal Data Bot"], check=True)

        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout

        if not status:
            logger.info("No changes detected. Skipping push.")
            return

        subprocess.run(["git", "add", "data/gold/voyages.parquet"], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto-update: {datetime.now().date()}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

        logger.info("Successfully pushed latest gold data to GitHub!")
    except Exception as e:
        logger.error(f"Sync failed: {e}")


@app.function(image=image, memory=1024, retries=3)
def compute_route_metrics(row: dict):
    from src.voyage_enrichment import compute_route_metrics

    cache = modal.Dict.from_name("maritime-distance-cache", create_if_missing=True)
    return compute_route_metrics(row, cache)


# --- MAIN PIPELINE ---


@app.function(image=image, volumes={"/data": volume}, memory=8192, timeout=3600)
def full_pipeline(year: int, month: int):
    logger.info(f"🚀 Starting Pipeline: {year}-{month:02d}")

    # Create the directories directly on the mounted Volume
    for folder in [SILVER_DIR, GOLD_DIR, REFERENCE_DIR, BRONZE_DIR]:
        os.makedirs(folder, exist_ok=True)

    # Commit the folder structure so subsequent workers see it
    volume.commit()

    # Force bootstrap for testing
    logger.info("Forcing bootstrap for testing...")
    run_unlocode_bootstrap.remote()

    while not os.path.exists(PORTS_PATH):
        logger.info("Waiting for PORTS_PATH to become visible...")
        time.sleep(5)
        volume.reload()

    # 1. Parallel Extraction via AISProcessor
    # Properly initialize the class with modal.parameter()
    processor = AISProcessor(model_name="voyage-v1")
    days_in_month = (
        (datetime(year, month % 12 + 1, 1) - timedelta(days=1)).day if month < 12 else 31
    )
    day_args = [(year, month, d) for d in range(1, days_in_month + 1)]

    logger.info(f"Extracting {days_in_month} days of AIS data...")
    # Using list() to force execution of the starmap generator
    results = list(processor.process_day.starmap(day_args, order_outputs=False))
    for r in results:
        logger.info(f"Worker report: {r}")

    volume.commit()  # Save the extracted Parquet files to volume

    # 2. Stitch Voyages (Transformation)
    logger.info("Stitching AIS pings into voyages...")
    stitch_voyages(year, month)
    volume.reload()

    # 3. Gold Layer Enrichment
    logger.info("Enriching voyages with maritime distances...")

    # Look for any silver parquet files for the month
    silver_files = list(SILVER_DIR.glob(f"ais_{year}_{month:02d}_*.parquet"))
    if not silver_files:
        logger.error(f"No silver files found for {year}-{month:02d}. Skipping enrichment.")
        return

    # Combine all silver files for the month
    silver_dfs = []
    for silver_file in silver_files:
        try:
            df = pl.read_parquet(str(silver_file))
            silver_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {silver_file}: {e}")

    if not silver_dfs:
        logger.error("No valid silver files found. Skipping enrichment.")
        return

    legs_df = pl.concat(silver_dfs)
    logger.info(f"Enriching {len(legs_df)} voyage legs with maritime distances...")

    gold_results = list(compute_route_metrics.map(legs_df.to_dicts(), order_outputs=False))
    gold_df = pl.DataFrame(gold_results)

    # Time and speed calculations
    if not gold_df.is_empty():
        # Parse timestamps and calculate duration
        gold_df = gold_df.with_columns(
            [
                pl.col("dep_time").str.to_datetime(strict=False).alias("dep_time_parsed"),
                pl.col("arr_time").str.to_datetime(strict=False).alias("arr_time_parsed"),
            ]
        )

        gold_df = gold_df.with_columns(
            [
                (
                    (pl.col("arr_time_parsed") - pl.col("dep_time_parsed")).dt.total_seconds()
                    / 3600
                ).alias("duration_hrs")
            ]
        )

        gold_df = gold_df.with_columns(
            [
                (pl.col("trip_distance_nm") / pl.col("duration_hrs"))
                .fill_nan(0)
                .alias("avg_speed_kts")
            ]
        )

        # Clean up temporary columns
        gold_df = gold_df.drop(["dep_time_parsed", "arr_time_parsed"])

        gold_path = f"/data/gold/voyages_{year}_{month:02d}.parquet"
        os.makedirs("/data/gold", exist_ok=True)
        gold_df.write_parquet(gold_path)
        logger.info(f"Saved gold data to {gold_path}")

    volume.commit()
    logger.info("🏁 Pipeline complete.")


# --- AUTOMATION & ENTRYPOINTS ---


@app.function(schedule=modal.Cron("0 2 * * *"))
def daily_update():
    now = datetime.now()
    # Run for previous month if today is early in the month, or current month
    # Here we default to current month for simplicity
    full_pipeline.remote(now.year, now.month)
    sync_to_github.remote()


@app.local_entrypoint()
def main(year: int = 2025, month: int = 1):
    full_pipeline.remote(year, month)
    sync_to_github.remote()
