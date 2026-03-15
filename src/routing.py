import modal
from typing import Optional, Tuple
from src.config import app, image

@app.function(image=image, cpu=1.0)
def compute_single_route(coords: Tuple[float, float, float, float]) -> Optional[float]:
    """Modal worker: Calculates distance for a single voyage segment."""
    from searoute import searoute
    try:
        route = searoute([coords[0], coords[1]], [coords[2], coords[3]], units="nm")
        return route['properties']['length']
    except Exception:
        return None
