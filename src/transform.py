import duckdb
from src.config import BRONZE_DIR, SILVER_DIR, get_logger

logger = get_logger(__name__)

def stitch_voyages(year: int, month: int):
    """Uses DuckDB window functions to pair departures and arrivals."""
    con = duckdb.connect()
    input_pattern = f"{BRONZE_DIR}/ais_{year}_{month:02d}*.parquet"
    output_path = SILVER_DIR / f"silver_{year}_{month:02d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Stitching Silver layer for {year}-{month:02d}...")
    try:
        con.execute(f"""
            COPY (
                WITH events AS (
                    SELECT mmsi, imo, vessel_name, base_date_time as dt,
                           port_locode, latitude as lat, longitude as lon,
                           LAG(port_locode) OVER (PARTITION BY imo ORDER BY base_date_time) as prev_port
                    FROM read_parquet('{input_pattern}')
                )
                SELECT * FROM events WHERE prev_port IS NULL OR port_locode != prev_port
            ) TO '{output_path}' (FORMAT 'PARQUET')
        """)
        logger.info(f"Silver layer saved: {output_path}")
    except Exception as e:
        logger.error(f"Silver stitching failed: {e}")
