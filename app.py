import modal
import duckdb
import polars as pl
from src.config import app, volume, image, SILVER_DIR, GOLD_DIR
from src.transform import run_silver_stitching
from src.routing import compute_single_route

@app.function(image=image, volumes={"/data": volume}, memory=4096, timeout=3600)
def etl_pipeline(year: int, month: int):
    print(f"🚀 Starting Medallion Pipeline for {year}-{month:02d}")
    
    # 1. Silver Layer (Assuming Bronze data is already downloaded to /data/bronze)
    run_silver_stitching(year, month)
    
    # 2. Gold Layer (Voyages + Routing)
    con = duckdb.connect()
    silver_file = SILVER_DIR / f"stitched_{year}_{month:02d}.parquet"
    
    # Filter to actual voyages (Port A to Port B)
    voyages_df = con.execute(f"""
        SELECT * FROM read_parquet('{silver_file}')
        WHERE port_locode != next_locode AND next_locode IS NOT NULL
    """).df()
    
    # 3. Parallel Routing via Modal workers
    print(f"🥇 Routing {len(voyages_df)} voyages in parallel...")
    route_tasks = list(zip(voyages_df['lon'], voyages_df['lat'], voyages_df['next_lon'], voyages_df['next_lat']))
    voyages_df['distance_nm'] = list(compute_single_route.map(route_tasks))
    
    # Calculate duration
    voyages_df['duration_hours'] = (voyages_df['next_dt'] - voyages_df['dt']).dt.total_seconds() / 3600
    
    # Persist Gold layer
    gold_file = GOLD_DIR / f"voyages_{year}_{month:02d}.parquet"
    gold_file.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(voyages_df).write_parquet(gold_file)
    
    volume.commit()
    print("✅ Pipeline Success.")

@app.function(image=image, volumes={"/data": volume}, allow_cross_origin_requests=True)
@modal.asgi_app()
def ui():
    import subprocess
    subprocess.Popen(["streamlit", "run", "dashboard.py", "--server.port", "8000", "--server.address", "0.0.0.0"])
