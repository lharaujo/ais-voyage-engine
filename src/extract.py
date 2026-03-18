import os
import re
from pathlib import Path
from typing import Optional

import modal
import numpy as np
import polars as pl
import requests
import zstandard as zstd
from bs4 import BeautifulSoup
from scipy.spatial import KDTree
from tqdm import tqdm

from src.config import PORTS_PATH, SILVER_DIR, app, get_logger, image, volume
from src.datetime_utils import parse_ais_datetime
from src.geospacial import to_radians

logger = get_logger(__name__)

# --- UTILS ---


def transform_coords_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Converts UN/LOCODE DMS format coordinates to Decimal Degrees.

    Args:
        df: DataFrame with 'Coordinates' column in DMS format (e.g., "5130N 00007W")

    Returns:
        DataFrame with added 'lat' and 'lon' columns in decimal degrees
    """

    def parse_lat(c: str):
        if not c or len(c.strip()) < 9:
            return None  # Need at least "DDMMH DDDMMH"
        try:
            parts = c.strip().split()
            if len(parts) != 2:
                return None
            lat_part = parts[0]
            if len(lat_part) < 5 or lat_part[-1] not in "NS":
                return None
            deg = float(lat_part[0:2])
            mnt = float(lat_part[2:4]) / 60
            if deg < 0 or deg > 90 or mnt < 0 or mnt >= 1:
                return None
            val = deg + mnt
            return -val if lat_part[-1] == "S" else val
        except Exception:
            return None

    def parse_lon(c: str):
        if not c or len(c.strip()) < 9:
            return None  # Need at least "DDMMH DDDMMH"
        try:
            parts = c.strip().split()
            if len(parts) != 2:
                return None
            lon_part = parts[1]
            if len(lon_part) < 6 or lon_part[-1] not in "EW":
                return None
            deg = float(lon_part[0:3])
            mnt = float(lon_part[3:5]) / 60
            if deg < 0 or deg > 180 or mnt < 0 or mnt >= 1:
                return None
            val = deg + mnt
            return -val if lon_part[-1] == "W" else val
        except Exception:
            return None

    return df.with_columns(
        [
            pl.col("Coordinates").map_elements(parse_lat, return_dtype=pl.Float64).alias("lat"),
            pl.col("Coordinates").map_elements(parse_lon, return_dtype=pl.Float64).alias("lon"),
        ]
    ).drop_nulls(["lat", "lon"])


# --- BOOTSTRAP ---


@app.function(volumes={"/data": volume}, timeout=1200)
def run_unlocode_bootstrap():
    os.makedirs(os.path.dirname(PORTS_PATH), exist_ok=True)
    if os.path.exists(PORTS_PATH):
        logger.info(f"✅ Reference data exists at {PORTS_PATH}")
        return

    logger.info("📖 Scraping ISO codes and LOCODE data...")
    headers = {"User-Agent": "Mozilla/5.0"}
    all_chunks = []

    # Simplified scraping logic for brevity in consolidation
    try:
        wiki_res = requests.get(
            "https://en.wikipedia.org/wiki/ISO_3166-2", headers=headers, timeout=15
        )
        soup = BeautifulSoup(wiki_res.text, "html.parser")
        iso_codes = sorted(
            {
                link["href"][-2:].lower()
                for link in soup.find_all("a", href=re.compile(r"/wiki/ISO_3166-2:[A-Z]{2}$"))
            }
        )

        for iso in tqdm(iso_codes, desc="Bootstrapping Ports"):
            res = requests.get(
                f"https://service.unece.org/trade/locode/{iso}.htm", headers=headers, timeout=10
            )
            if res.status_code != 200:
                continue

            s = BeautifulSoup(res.text, "html.parser")
            rows = []
            for tr in s.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if len(cells) >= 10:
                    rows.append(
                        {
                            "CountryCode": iso.upper(),
                            "LOCODE": "".join(cells[1].split()),
                            "Name": cells[2],
                            "Function": cells[5],
                            "Status": cells[6].strip()[:2],
                            "Coordinates": cells[9].strip(),
                        }
                    )

            if rows:
                df = pl.DataFrame(rows).filter(
                    (pl.col("Function").str.contains("1"))
                    & (pl.col("Coordinates").str.contains(r"\d{4}[NS]"))
                    & (
                        pl.col("Status").is_in(
                            ["AA", "AC", "AF", "AI", "AM", "AS", "AQ", "RL", "RQ", "RN", "QQ"]
                        )
                    )
                )
                if not df.is_empty():
                    all_chunks.append(
                        transform_coords_polars(df).select(
                            ["CountryCode", "LOCODE", "Name", "lat", "lon"]
                        )
                    )

        if all_chunks:
            final_df = pl.concat(all_chunks).unique(subset=["LOCODE"])
            final_df.write_parquet(PORTS_PATH)
            volume.commit()
            logger.info(f"💾 Saved {len(final_df)} ports.")
    except Exception as e:
        logger.error(f"❌ Bootstrap failed: {e}")


@app.function(volumes={"/data": volume})
def test_ports_file():
    """Test function to check if ports file can be read."""
    try:
        df = pl.read_parquet(PORTS_PATH)
        logger.info(f"✅ Successfully read {len(df)} ports")
        return f"Success: {len(df)} rows"
    except Exception as e:
        logger.error(f"❌ Failed to read ports file: {e}")
        return f"Error: {e}"


@app.cls(image=image, volumes={"/data": volume}, memory=4096, timeout=600)
class AISProcessor:
    model_name: str = modal.parameter()
    batch_size: int = modal.parameter(default=100)
    save_path: str = modal.parameter(default=str(SILVER_DIR))

    # Class-level defaults - properly typed
    tree: Optional[KDTree] = None
    port_locodes: Optional[np.ndarray] = None
    port_names: Optional[np.ndarray] = None

    def __enter__(self):
        """Initialize the KDTree for port proximity calculations."""
        try:
            # Read reference data
            df = pl.read_parquet(PORTS_PATH)

            # Initialize shared resources
            self.port_locodes = df["LOCODE"].to_numpy()
            self.port_names = df["Name"].to_numpy()
            coords = np.deg2rad(df[["lat", "lon"]].to_numpy())
            self.tree = KDTree(coords)

            logger.info(f"✅ Worker {os.getpid()} initialized successfully.")
            return
        except Exception as e:
            logger.error(f"❌ Worker failed to initialize: {e}")
            self.tree = None

    @modal.method()
    def process_day(self, year: int, month: int, day: int) -> str:
        # Try to initialize KDTree if not already done
        if self.tree is None:
            try:
                df = pl.read_parquet(PORTS_PATH)
                self.port_locodes = df["LOCODE"].to_numpy()
                self.port_names = df["Name"].to_numpy()
                coords = np.deg2rad(df[["lat", "lon"]].to_numpy())
                self.tree = KDTree(coords)
                logger.info(f"✅ Worker {os.getpid()} initialized KDTree for day {day}")
            except Exception as e:
                return (
                    f"❌ FAILED: Could not initialize KDTree for {year}-{month:02d}-{day:02d}: {e}"
                )

        date_str = f"{year}-{month:02d}-{day:02d}"
        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/ais-{date_str}.csv.zst"

        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            import io

            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(resp.raw) as reader:
                csv_bytes = reader.read()
                csv_buffer = io.BytesIO(csv_bytes)

                # Read once with schema inspection
                ais = pl.read_csv(csv_buffer, ignore_errors=True)

                # Select only available columns
                relevant_cols = [
                    col
                    for col in [
                        "mmsi",
                        "latitude",
                        "longitude",
                        "base_date_time",
                        "vessel_name",
                        "imo",
                    ]
                    if col in ais.columns
                ]
                ais = ais.select(relevant_cols).filter(pl.col("mmsi").is_not_null())
                ais = ais.with_columns(
                    [parse_ais_datetime(pl.col("base_date_time")).alias("dt")]
                ).drop("base_date_time")

            # Ensure latitude and longitude columns exist
            if "latitude" in ais.columns and "longitude" in ais.columns:
                ais_valid = ais.filter(
                    pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
                )
                if ais_valid.is_empty():
                    return f"No valid lat/lon data for {date_str}"

                # Convert coordinates to radians for KDTree
                ship_coords = to_radians(ais_valid)
                self.tree = KDTree(to_radians(df, "lat", "lon"))
                dist, idx = self.tree.query(ship_coords, k=1)
                mask = (dist * 6371.0) <= 5.0  # 5km radius

                if mask.any():
                    valid_idx = idx[mask]
                    # Ensure valid_idx is within bounds
                    max_idx = min(len(self.port_locodes), len(self.port_names))
                    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < max_idx)]
                    if len(valid_idx) == 0:
                        return f"No valid port indices for {date_str}"
                    filtered_ais = ais_valid.filter(mask)
                    filtered_ais = filtered_ais.head(len(valid_idx))  # Ensure same length
                    res = filtered_ais.with_columns(
                        [
                            pl.Series("port_locode", self.port_locodes[valid_idx]),
                            pl.Series("port_name", self.port_names[valid_idx]),
                        ]
                    ).rename({"latitude": "lat", "longitude": "lon"})

                    out_path = Path(self.save_path) / f"ais_{year}_{month:02d}_{day:02d}.parquet"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    res.write_parquet(out_path)
                    return f"✅ Saved {len(res)} rows to {out_path}"

                return f"No port pings for {date_str}"
            else:
                logger.warning(f"Invalid data for {date_str}: Missing latitude/longitude columns")
                return f"No valid lat/lon data for {date_str}"

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            return f"Error on {date_str}: {e}"
