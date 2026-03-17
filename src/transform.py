import duckdb

from src.config import SILVER_DIR, get_logger

logger = get_logger(__name__)


def stitch_voyages(year: int, month: int):
    con = duckdb.connect()
    input_pattern = f"{SILVER_DIR}/ais_{year}_{month:02d}*.parquet"
    output_path = SILVER_DIR / f"silver_{year}_{month:02d}.parquet"

    logger.info(f"Stitching Silver layer for {year}-{month:02d}...")
    try:
        con.execute(
            f"""
            COPY (
                WITH unique_port_visits AS (
                    SELECT
                        mmsi,
                        REGEXP_REPLACE(imo, '^IMO0*', '') as imo,
                        vessel_name,
                        dt,
                        port_locode,
                        lat,
                        lon,
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
                        port_locode as dep_locode,
                        dt as dep_time,
                        lat as dep_lat,
                        lon as dep_lon,
                        LEAD(port_locode) OVER (PARTITION BY mmsi ORDER BY dt) as arr_locode,
                        LEAD(dt) OVER (PARTITION BY mmsi ORDER BY dt) as arr_time,
                        LEAD(lat) OVER (PARTITION BY mmsi ORDER BY dt) as arr_lat,
                        LEAD(lon) OVER (PARTITION BY mmsi ORDER BY dt) as arr_lon
                    FROM filtered_events
                )
                -- Final Select where we can safely filter the window function results
                SELECT * FROM voyage_legs
                WHERE arr_locode IS NOT NULL
            ) TO '{output_path}' (FORMAT 'PARQUET')
        """
        )
        logger.info(f"Silver layer saved: {output_path}")
    except Exception as e:
        logger.error(f"Silver stitching failed: {e}")
        raise
