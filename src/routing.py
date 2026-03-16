from searoute import searoute

from src.config import app, get_logger, image

logger = get_logger(__name__)


@app.function(image=image, cpu=1.0)
def compute_distance(coords: tuple):
    """
    Calculates maritime distance between two points.
    Input: (lon1, lat1, lon2, lat2)
    """
    try:
        route = searoute([coords[0], coords[1]], [coords[2], coords[3]], units="nm")
        return route["properties"]["length"]
    except Exception as e:
        logger.warning(f"Routing failed for {coords}: {e}")
        return None
