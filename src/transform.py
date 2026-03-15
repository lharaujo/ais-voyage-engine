import duckdb
from src.config import BRONZE_DIR, SILVER_DIR

def run_silver_stitching(year: int, month: int):
    """Aligns pings into Departure/Arrival events."""
    con = duckdb.connect()
    
    output_path = SILVER_DIR / f"stitched_{year}_{month:02d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    con.execute(f"""
        COPY (
            SELECT 
                mmsi, imo, vessel_name, vessel_type,
                base_date_time::TIMESTAMP as dt,
                port_locode, lat, lon,
                LEAD(base_date_time::TIMESTAMP) OVER w as next_dt,
                LEAD(port_locode) OVER w as next_locode,
                LEAD(lat) OVER w as next_lat,
                LEAD(lon) OVER w as next_lon
            FROM read_parquet('{BRONZE_DIR}/ais_{year}_{month:02d}*.parquet')
            WINDOW w AS (PARTITION BY imo ORDER BY base_date_time)
        ) TO '{output_path}' (FORMAT 'PARQUET')
    """)
    print(f"🥈 Silver layer stitched: {output_path}")
