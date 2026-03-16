import polars as pl
import pytest

from src.extract import transform_coords_polars


def test_coordinate_transformation():
    """Test that UN/LOCODE strings are correctly converted to decimal degrees."""
    # 1. Create a dummy DataFrame mimicking the scraped Wikipedia/UNECE data
    # Format: DDMM[NS] DDDMM[EW]
    data = {
        "Coordinates": [
            "5130N 00007W",  # London
            "4043N 07400W",  # New York
            "3355S 01825E",  # Cape Town
            "3541N 13946E",  # Tokyo
        ]
    }
    df = pl.DataFrame(data)

    # 2. Run the transformation
    result_df = transform_coords_polars(df)

    # 3. Assertions (Expected Decimal Degrees)
    # London: 51 + 30/60 = 51.5 | -0.1166 (approx)
    london = result_df.row(0, named=True)
    assert pytest.approx(london["lat"], 0.1) == 51.5
    assert london["lon"] < 0  # Should be West (negative)

    # Cape Town: Should be negative Latitude (South)
    cape_town = result_df.row(2, named=True)
    assert cape_town["lat"] < 0
    assert cape_town["lon"] > 0
