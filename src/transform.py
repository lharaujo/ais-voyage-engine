from typing import Optional

import duckdb
import polars as pl

from datetime_utils import parse_ais_datetime, time_difference_hours
from src.config import GOLD_DIR, SILVER_DIR, get_logger
from src.settings import AppSettings
from src.voyage_enrichment import generate_sea_path

logger = get_logger(__name__)


def stitch_voyages(year: int, month: int, settings: Optional[AppSettings] = None) -> None:
    """
    Stitch AIS pings into voyage legs using DuckDB window functions.

    Args:
        year: Year (YYYY)
        month: Month (1-12)
        settings: Application configuration

    Returns:
        None (writes parquet to gold layer)
    """

    settings = settings or AppSettings()
    input_pattern = SILVER_DIR / f"ais_{year}_{month:02d}*.parquet"
    output_path = GOLD_DIR / "voyages.parquet"

    logger.info(f"🧵 Stitching Silver layer for {year}-{month:02d}...")

    sql = f"""
        WITH unique_port_visits AS (
            SELECT
                mmsi,
                dt AS dep_time,
                dt AS arr_time,
                REGEXP_REPLACE(imo, '^IMO', '') as imo,
                vessel_name, dt, port_locode, lat, lon,
                LAG(port_locode) OVER (PARTITION BY mmsi ORDER BY dt) as prev_port
            FROM read_parquet('{input_pattern}')
        ),
        filtered_events AS (
            SELECT * FROM unique_port_visits
            WHERE prev_port IS NULL OR port_locode != prev_port
        ),
        voyage_legs AS (
            SELECT
                mmsi, imo, vessel_name,
                port_locode as dep_locode, lat as dep_lat, lon as dep_lon,
                LEAD(port_locode) OVER (PARTITION BY mmsi ORDER BY dt) as arr_locode,
                LEAD(lat) OVER (PARTITION BY mmsi ORDER BY dt) as arr_lat,
                LEAD(lon) OVER (PARTITION BY mmsi ORDER BY dt) as arr_lon,
                LEAD(dt) OVER (PARTITION BY mmsi ORDER BY dt) - dt as duration
            FROM filtered_events
        )
        SELECT
            mmsi, imo, vessel_name,
            dep_locode, arr_locode, dep_lat, dep_lon, arr_lat, arr_lon
        FROM voyage_legs
        WHERE arr_locode IS NOT NULL
    """

    try:
        with duckdb.connect() as con:
            df = con.execute(sql).pl()

        if df.is_empty():
            logger.warning("No voyages found to process.")
            return

        df = df.with_columns(
            [
                parse_ais_datetime(pl.col("arr_time")).alias("arr_time"),
                parse_ais_datetime(pl.col("dep_time")).alias("dep_time"),
            ]
        )
        df = df.with_columns(
            [time_difference_hours(pl.col("arr_time"), pl.col("dep_time")).alias("duration_hrs")]
        )

        logger.info(f"🛰️ Generating optimized sea paths for {len(df)} voyages...")
        df = df.with_columns(
            [
                pl.struct(["dep_lon", "dep_lat", "arr_lon", "arr_lat"])
                .map_elements(
                    lambda row: generate_sea_path(
                        {
                            "dep_lon": row["dep_lon"],
                            "dep_lat": row["dep_lat"],
                            "arr_lon": row["arr_lon"],
                            "arr_lat": row["arr_lat"],
                        }
                    ),
                    return_dtype=pl.List(pl.List(pl.Float64)),
                )
                .alias("path")
            ]
        )

        df = df.with_columns(
            [
                pl.col("path")
                .map_elements(lambda x: len(x) * 12.0, return_dtype=pl.Float64)
                .alias("trip_distance_nm"),
                pl.lit(12.5).alias("avg_speed_kts"),
            ]
        )

        df = df.drop(["dep_lat", "dep_lon", "arr_lat", "arr_lon"])

        GOLD_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, compression="snappy")

        logger.info(f"✅ Gold layer saved: {output_path}")

    except Exception as e:
        logger.error(f"❌ Transformation failed: {e}")
        raise
