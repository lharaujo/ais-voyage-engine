import calendar
import os

import modal
import polars as pl

from src.config import SILVER_DIR, app, get_logger, image, volume

# ADDED: run_unlocode_bootstrap added to imports
from src.extract import process_daily_ais, run_unlocode_bootstrap

# Ensure these exist in your transform.py
from src.transform import stitch_voyages

logger = get_logger("orchestrator")

# Create a persistent dictionary named "port-distances"
# This survives even after the app stops running
distance_lookup = modal.Dict.from_name("port-distances", create_if_missing=True)
distance_cache = modal.Dict.from_name("maritime-distance-cache", create_if_missing=True)


@app.function(image=image, volumes={"/data": volume}, memory=8192, timeout=3600)
def full_pipeline(year: int, month: int):
    logger.info(f"🚀 Starting Pipeline: {year}-{month:02d}")

    # --- START OF BOOTSTRAP LOGIC ---
    # Check if reference data exists before starting extraction
    ports_path = "/data/reference/ports.parquet"
    if not os.path.exists(ports_path):
        logger.warning("⚠️ Reference data (ports.parquet) missing! Initializing bootstrap...")
        # Trigger the scraper we adapted in extract.py
        # Using .remote() ensures it runs in its own optimized container context
        run_unlocode_bootstrap.remote()

        # Reload the volume to ensure this worker sees the newly created file
        volume.reload()
        logger.info("✅ Reference data successfully bootstrapped.")
    else:
        logger.info("⚓ Reference data verified.")
    # --- END OF BOOTSTRAP LOGIC ---

    # 1. Parallel Extraction
    days = range(1, calendar.monthrange(year, month)[1] + 1)
    day_args = [(year, month, d) for d in days]
    for _ in extract_worker.map(day_args, order_outputs=False):
        pass

    volume.reload()

    # 2. Stitching (Silver Layer)
    logger.info("🧵 Stitching daily files into voyage legs...")
    stitch_voyages(year, month)
    volume.reload()

    # 3. Maritime Routing (Gold Layer)
    logger.info("🛰️ Calculating maritime distances via searoute...")
    silver_path = SILVER_DIR / f"silver_{year}_{month:02d}.parquet"
    legs_df = pl.read_parquet(str(silver_path))

    # Map the compute_route_metrics function across workers
    chunk_size = 5000
    gold_results = []

    try:
        for result in compute_route_metrics.map(legs_df.to_dicts(), order_outputs=False):
            gold_results.append(result)
            if len(gold_results) % chunk_size == 0:
                logger.info(f"📥 Collected {len(gold_results)} / {len(legs_df)} results...")
    except Exception as e:
        logger.error(f"❌ Map interrupted: {e}")

    # Save the Gold result
    gold_df = pl.DataFrame(gold_results)

    # Perform all calculations in a single clean chain
    gold_df = gold_df.with_columns(
        [
            # Convert strings to datetime using the specific format to avoid errors
            (
                (
                    pl.col("arrival_time").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
                    - pl.col("departure_time").str.to_datetime(
                        format="%Y-%m-%d %H:%M:%S", strict=False
                    )
                ).dt.total_seconds()
                / 3600
            ).alias("duration_hrs")
        ]
    ).with_columns(
        [
            # Use the duration we just calculated to get speed
            (pl.col("trip_distance_nm") / pl.col("duration_hrs"))
            .fill_nan(0)
            .alias("avg_speed_kts")
        ]
    )

    gold_path = f"/data/gold/voyages_{year}_{month:02d}.parquet"
    os.makedirs("/data/gold", exist_ok=True)
    if os.path.exists(gold_path):
        logger.info(f"⏩ Gold file already exists for {year}-{month}, skipping routing.")
        return
    gold_df.write_parquet(gold_path)

    volume.commit()
    logger.info(f"✅ Gold file saved: {gold_path}")

    # Export the cache to a persistent Parquet file for manual analysis
    logger.info("💾 Exporting distance cache to Parquet...")

    try:
        # 1. Reference the dict by name directly
        # 'create_if_missing=False' is safer here since we're just reading
        cache = modal.Dict.from_name("maritime-distance-cache", create_if_missing=True)
        cache_items = list(cache.items())

        if cache_items:
            # 2. Convert to Polars and save to Volume
            cache_df = pl.DataFrame([{"route_key": k, "dist_km": v} for k, v in cache_items])

            cache_export_path = "/data/reference/port_distances.parquet"
            os.makedirs("/data/reference", exist_ok=True)
            cache_df.write_parquet(cache_export_path)

            logger.info(
                f"✅ Cache backed up: {len(cache_items)} routes saved to {cache_export_path}"
            )
        else:
            logger.warning("⚠️ Cache is empty; nothing to export.")

    except Exception as e:
        logger.error(f"❌ Failed to export cache: {e}")

    # 3. Final commit to make sure the Parquet file and gold data are persisted
    volume.commit()
    logger.info("🏁 Pipeline complete and volume committed.")


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=4096,
    max_containers=5,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=5.0,
    ),
)
def extract_worker(args):
    year, month, day = args
    process_daily_ais(year, month, day)


@app.local_entrypoint()
def main(year: int = 2025, month: int = 1):
    full_pipeline.remote(year, month)


@app.function(image=image, memory=1024, retries=3)
def compute_route_metrics(row):
    import modal  # Imported inside to ensure it's available in the container
    from searoute import searoute

    # 1. Safely grab the persistent dictionary by name
    # This ensures the worker connects to the correct shared storage
    distance_cache = modal.Dict.from_name("maritime-distance-cache", create_if_missing=True)

    origin_code = row["dep_locode"]
    dest_code = row["arr_locode"]
    cache_key = f"{origin_code}_{dest_code}"

    dist_km = 0.0

    try:
        # 2. Check if we already have this distance
        if cache_key in distance_cache:
            dist_km = distance_cache[cache_key]
        else:
            # 3. Call searoute if missing
            origin_coords = [row["dep_lon"], row["dep_lat"]]
            dest_coords = [row["arr_lon"], row["arr_lat"]]

            route_geo = searoute(origin_coords, dest_coords, units="km")
            dist_km = route_geo["properties"]["length"]

            # 4. Save back to the persistent cache
            distance_cache[cache_key] = dist_km

    except Exception:
        # Fallback if searoute or cache fails
        dist_km = 0.0

    # 5. Derive Nautical Miles mathematically (1 NM = 1.852 KM)
    dist_nm = dist_km / 1.852 if dist_km > 0 else 0.0

    return {
        "mmsi": row["mmsi"],
        "imo": row["imo"],
        "ship_name": row.get("vessel_name", "Unknown"),
        "locode_departure": origin_code,
        "locode_arrival": dest_code,
        "lat_departure": row["dep_lat"],
        "lon_departure": row["dep_lon"],
        "lat_arrival": row["arr_lat"],
        "lon_arrival": row["arr_lon"],
        "departure_time": row["dep_time"],
        "arrival_time": row["arr_time"],
        "trip_distance_km": dist_km,
        "trip_distance_nm": dist_nm,
    }
