import polars as pl
import numpy as np
import requests
import zstandard as zstd
import io
import tempfile
import os
from scipy.spatial import cKDTree
from bs4 import BeautifulSoup
from src.config import UNLOCODE_PATH, BRONZE_DIR, get_logger

logger = get_logger(__name__)

def transform_coords_polars(df: pl.DataFrame) -> pl.DataFrame:
   """
    Parses UN/LOCODE geographic coordinates from DDMM[N/S] DDDMM[E/W] format to Decimal Degrees.
    
    Args:
        df (pl.DataFrame): Input dataframe containing a 'Coordinates' column.
        
    Returns:
        pl.DataFrame: Dataframe with 'lat' and 'lon' columns in float format.
    """
    coord_regex = r"^(\d{2})(\d{2})([NS])\s*(\d{1,3})(\d{2})([EW]).*$"
    return df.with_columns(
        lat_deg = pl.col("Coordinates").str.extract(coord_regex, 1).cast(pl.Float64),
        lat_min = pl.col("Coordinates").str.extract(coord_regex, 2).cast(pl.Float64),
        lat_dir = pl.col("Coordinates").str.extract(coord_regex, 3),
        lon_deg = pl.col("Coordinates").str.extract(coord_regex, 4).cast(pl.Float64),
        lon_min = pl.col("Coordinates").str.extract(coord_regex, 5).cast(pl.Float64),
        lon_dir = pl.col("Coordinates").str.extract(coord_regex, 6)
    ).with_columns(
        lat = (pl.col("lat_deg") + (pl.col("lat_min") / 60)) * pl.when(pl.col("lat_dir") == "S").then(-1.0).otherwise(1.0),
        lon = (pl.col("lon_deg") + (pl.col("lon_min") / 60)) * pl.when(pl.col("lon_dir") == "W").then(-1.0).otherwise(1.0)
    ).drop(["lat_deg", "lat_min", "lat_dir", "lon_deg", "lon_min", "lon_dir"])

def run_unlocode_scraper():
   """
    Scrapes the UNECE website for UN/LOCODE port data and saves it as a reference Parquet.
    
    This function navigates the Wikipedia ISO list to find individual country pages,
    parses HTML tables for ports with functional status '1', and converts coordinates.
    """
    url = "https://en.wikipedia.org/wiki/ISO_3166-2"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a', href=re.compile(r"/wiki/ISO_3166-2:[A-Z]{2}$"))
        codes = {link['href'][-2:].lower() for link in links}
        codes_list = sorted(list(codes))
        logger.info(f"Success Found {len(codes_list)} country codes.")
        return codes_list
    except Exception as e:
        logger.critical(f"Unexpected failure downloading country codes: {e}", exec_info=True)
        return [] 

def process_daily_ais(year, month, day):
    """
    Downloads, decompresses, and spatially filters a single day of NOAA AIS data.
    
    Utilizes a cKDTree for 5km geofencing around ports. This function is memory-optimized
    to run within a 4GB constraint by using Polars Lazy scanning and streaming decompression.
    
    Args:
        year (int): The year of the AIS data.
        month (int): The month of the AIS data.
        day (int): The day of the AIS data.
    """
    try:
        ports_df = pl.read_parquet(UNLOCODE_PATH)
        logger.info("Success UNLOCODE loaded")
    except Exception e:
        logger.critical(f"Error while loading UNLOCODE: {e}")
    port_tree = cKDTree(np.deg2rad(ports_df.select(["lat", "lon"]).to_numpy()))

    date_str = f"{year}-{month:02d}-{day:02d}"
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/ais-{date_str}.csv.zst"

    try:
     resp = requests.get(url, stream=True, timeout=60)
     resp.status_code == 200
     logger.info(f"Success ais-{date_str} downloaded")      return
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error for {year}-{month}-{day}: {e}")
    except Exception as e:
        logger.critical(f"Unexpected failure downloading AIS data: {e}", exc_info=True)
    # Decompress and filter in one pass using Polars Lazy
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(resp.raw) as reader:
            while chunk := reader.read(1024 * 1024):
                tmp.write(chunk)
        
        ais = (
            pl.scan_csv(tmp.name, ignore_errors=True)
            .filter(pl.col("imo").is_not_null())
            .collect()
        )
        
        # Spatial Masking
        ship_coords = np.deg2rad(ais.select(["latitude", "longitude"]).to_numpy())
        dist, idx = port_tree.query(ship_coords, k=1)
        mask = dist * 6371.0 <= 5.0
        
        if mask.any():
            res = ais.filter(mask).with_columns([
                pl.Series("port_locode", ports_df["LOCODE"].to_numpy()[idx[mask]]),
                pl.Series("port_name", ports_df["Name"].to_numpy()[idx[mask]])
            ])
            out_path = BRONZE_DIR / f"ais_{year}_{month:02d}_{day:02d}.parquet"
            res.write_parquet(out_path)
