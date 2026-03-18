from typing import Dict, List

import searoute

from src.config import get_logger
from src.constants import (
    COORDINATE_PRECISION_DECIMALS,
    KM_TO_NM_CONVERSION,
    PATH_DOWNSAMPLING_CONFIG,
)

logger = get_logger(__name__)


def generate_sea_path(row: Dict[str, float]) -> List[List[float]]:
    """
    Generate optimized maritime path via searoute API.

    Args:
        row: Voyage data with dep_lon, dep_lat, arr_lon, arr_lat

    Returns:
        List of [lon, lat] coordinates, downsampled for file size

    Raises:
        Returns straight-line path if searoute fails (logged as warning)
    """
    origin = [row["dep_lon"], row["dep_lat"]]
    destination = [row["arr_lon"], row["arr_lat"]]

    try:
        route = searoute.searoute(origin, destination, append_orig_dest=True)
        coords = route.get("geometry", {}).get("coordinates", [])

        if not coords:
            return _fallback_path(row)

        # Minify coordinates
        minified = [
            [
                round(pt[0], COORDINATE_PRECISION_DECIMALS),
                round(pt[1], COORDINATE_PRECISION_DECIMALS),
            ]
            for pt in coords
        ]

        # Downsample if too many points
        if len(minified) > PATH_DOWNSAMPLING_CONFIG["long"]["threshold"]:
            return minified[:: PATH_DOWNSAMPLING_CONFIG["long"]["factor"]]
        elif len(minified) > PATH_DOWNSAMPLING_CONFIG["medium"]["threshold"]:
            return minified[:: PATH_DOWNSAMPLING_CONFIG["medium"]["factor"]]

        return minified

    except Exception as e:
        logger.warning(f"Searoute failed: {e}")
        return _fallback_path(row)


def _fallback_path(row: Dict[str, float]) -> List[List[float]]:
    """Generate straight-line path as fallback."""
    return [
        [
            round(row["dep_lon"], COORDINATE_PRECISION_DECIMALS),
            round(row["dep_lat"], COORDINATE_PRECISION_DECIMALS),
        ],
        [
            round(row["arr_lon"], COORDINATE_PRECISION_DECIMALS),
            round(row["arr_lat"], COORDINATE_PRECISION_DECIMALS),
        ],
    ]


def compute_route_metrics(row: dict, distance_cache: dict) -> dict:
    """Enriches voyage row with maritime distance metrics."""
    origin_code = row["dep_locode"]
    dest_code = row["arr_locode"]
    cache_key = f"{origin_code}_{dest_code}"

    if cache_key in distance_cache:
        dist_km = distance_cache[cache_key]
    else:
        try:
            origin = [row["dep_lon"], row["dep_lat"]]
            dest = [row["arr_lon"], row["arr_lat"]]
            route_geo = searoute.searoute(origin, dest, units="km")
            dist_km = route_geo["properties"]["length"]
            distance_cache[cache_key] = dist_km
        except Exception as e:
            logger.warning(f"Distance calc failed for {cache_key}: {e}")
            dist_km = 0.0

    return {
        **row,
        "trip_distance_km": dist_km,
        "trip_distance_nm": dist_km / KM_TO_NM_CONVERSION if dist_km > 0 else 0.0,
    }
