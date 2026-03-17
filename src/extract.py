import os
import re
import tempfile

import polars as pl
import requests
import zstandard as zstd
from bs4 import BeautifulSoup
from scipy.spatial import KDTree
from tqdm import tqdm

from src.config import PORTS_PATH, SILVER_DIR, app, get_logger, volume

logger = get_logger(__name__)


def transform_coords_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Converts UN/LOCODE DMS format (e.g., '4830N 01210E') to Decimal Degrees.
    """

    def parse_lat(c):
        if not c or len(c) < 5:
            return None
        try:
            deg = float(c[0:2])
            min = float(c[2:4]) / 60
            val = deg + min
            return -val if "S" in c else val
        except Exception:
            return None

    def parse_lon(c):
        # UN/LOCODE longitude can be 5 or 6 digits depending on leading zeros
        if not c or " " not in c:
            return None
        part = c.split(" ")[1]
        try:
            # Longitude is usually after the space: 01210E
            deg = float(part[0:3])
            min = float(part[3:5]) / 60
            val = deg + min
            return -val if "W" in part else val
        except Exception:
            return None

    # Apply transformations (Simplified for Polars map_elements or separate columns)
    # For performance in Polars, we use a regex extract if possible,
    # but UN/LOCODE strings are messy, so we use a clean map:
    return df.with_columns(
        [
            pl.col("Coordinates").map_elements(parse_lat, return_dtype=pl.Float64).alias("lat"),
            pl.col("Coordinates").map_elements(parse_lon, return_dtype=pl.Float64).alias("lon"),
        ]
    ).drop_nulls(["lat", "lon"])


def get_iso_codes_from_wikipedia():
    logger.info("📖 Scraping ISO 3166-2 Wikipedia page...")
    url = "https://en.wikipedia.org/wiki/ISO_3166-2"
    headers = {"User-Agent": "AIS-Voyage-Engine-Bot/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=re.compile(r"/wiki/ISO_3166-2:[A-Z]{2}$"))
        codes = {link["href"][-2:].lower() for link in links}
        return sorted(list(codes))
    except Exception as e:
        logger.error(f"❌ Wikipedia scrape failed: {e}")
        return []


@app.function(volumes={"/data": volume}, timeout=1200)
def run_unlocode_bootstrap():
    """
    The main entry point to ensure ports.parquet exists in the Modal Volume.
    """
    # Ensure the directory exists inside the volume
    os.makedirs(os.path.dirname(PORTS_PATH), exist_ok=True)

    if os.path.exists(PORTS_PATH):
        logger.info(f"✅ Reference data already exists at {PORTS_PATH}")
        return

    iso_codes = get_iso_codes_from_wikipedia()
    if not iso_codes:
        logger.error("Abort: No ISO codes found.")
        return

    logger.info(f"🌐 Bootstrap: Downloading UN/LOCODE data for {len(iso_codes)} countries...")
    all_chunks = []
    active_status = ["AA", "AC", "AF", "AI", "AM", "AS", "AQ", "RL", "RQ", "RN", "QQ"]
    headers = {"User-Agent": "Mozilla/5.0"}

    for iso in tqdm(iso_codes):
        try:
            url = f"https://service.unece.org/trade/locode/{iso}.htm"
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            rows = []
            for table in soup.find_all("table"):
                for tr in table.find_all("tr"):
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

            if not rows:
                continue

            df = pl.DataFrame(rows)
            # Filter for Ports (Function contains '1')
            df = df.filter(
                (pl.col("Function").str.contains("1"))
                & (pl.col("Coordinates").str.contains(r"\d{4}[NS]"))  # Basic DMS check
                & (pl.col("Status").is_in(active_status))
            )

            if not df.is_empty():
                df = transform_coords_polars(df).select(
                    ["CountryCode", "LOCODE", "Name", "lat", "lon"]
                )
                all_chunks.append(df)
        except Exception:
            continue

    if all_chunks:
        final_df = pl.concat(all_chunks).unique(subset=["LOCODE"])
        final_df.write_parquet(PORTS_PATH)
        volume.commit()  # Critical for Modal Volumes
        logger.info(f"💾 SUCCESS: Saved {len(final_df)} ports to {PORTS_PATH}")
    else:
        logger.error("❌ FAILED: No ports found during bootstrap.")


def process_daily_ais(year: int, month: int, day: int):
    """
    Downloads and spatially filters AIS data within 5km of ports.
    """
    import numpy as np

    try:
        # 1. Load Reference Data
        if not os.path.exists(PORTS_PATH):
            logger.error(f"❌ Reference file missing: {PORTS_PATH}")
            return

        ports_df = pl.read_parquet(PORTS_PATH)
        # Convert Lat/Lon to Radians for spherical distance calculation
        port_coords_rad = np.deg2rad(ports_df.select(["lat", "lon"]).to_numpy())
        port_tree = KDTree(port_coords_rad)

        logger.info(f"✅ UNLOCODE reference loaded: {len(ports_df)} ports.")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize Port Tree: {e}")
        return

    date_str = f"{year}-{month:02d}-{day:02d}"
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/ais-{date_str}.csv.zst"

    try:
        # 2. Streaming Download
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        logger.info(f"📥 Download started: {date_str}")

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            dctx = zstd.ZstdDecompressor()
            # Decompress directly to temporary file to save memory
            with dctx.stream_reader(resp.raw) as reader:
                while chunk := reader.read(1024 * 1024):
                    tmp.write(chunk)
            tmp.flush()

            # 3. Process with Polars
            # Using try/except here because CSVs can occasionally be malformed
            ais = (
                pl.scan_csv(tmp.name, ignore_errors=True)
                .filter(pl.col("mmsi").is_not_null())
                .collect()
            )

            if ais.is_empty():
                logger.warning(f"⚠️ No data found for {date_str}")
                return

            # 4. Spatial Masking (The KDTree Logic)
            ship_coords = np.deg2rad(ais.select(["latitude", "longitude"]).to_numpy())

            # Query the tree for the single nearest port (k=1)
            # dist is in radians because input was in radians
            dist, idx = port_tree.query(ship_coords, k=1)

            # Convert Radians to KM (Earth Radius approx 6371km)
            # Mask identifying pings within 5km of a port
            mask = (dist * 6371.0) <= 5.0

            if mask.any():
                # Filter AIS data and join with port metadata
                # We use the 'idx' from the tree to look up the Port names
                valid_indices = idx[mask]

                res = (
                    ais.filter(mask)
                    .with_columns(
                        [
                            pl.Series("port_locode", ports_df["LOCODE"].to_numpy()[valid_indices]),
                            pl.Series("port_name", ports_df["Name"].to_numpy()[valid_indices]),
                        ]
                    )
                    .rename({"latitude": "lat", "longitude": "lon", "base_date_time": "dt"})
                )

                # 5. Save to Silver Layer
                out_path = SILVER_DIR / f"ais_{year}_{month:02d}_{day:02d}.parquet"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                res.write_parquet(out_path)
                logger.info(f"💾 Silver file saved: {out_path} ({len(res)} rows)")
            else:
                logger.info(f"ℹ️ No pings found within 5km of ports for {date_str}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed for {date_str}: {e}")
