from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.settings import AppSettings, PathConfig
from src.transform import stitch_voyages


class TestVoyageStitching:
    """Test voyage stitching functionality."""

    @patch("src.transform.duckdb.connect")
    @patch("src.config.GOLD_DIR", Path("/tmp/test_gold"))
    @patch("src.config.SILVER_DIR", Path("/tmp/test_silver"))
    def test_stitch_voyages_success(self, mock_duckdb_connect, tmp_path):
        """Test successful voyage stitching."""
        # Mock the database connection and result
        mock_con = MagicMock()
        mock_duckdb_connect.return_value.__enter__.return_value = mock_con

        # Create mock DataFrame result
        mock_df = pl.DataFrame(
            {
                "mmsi": [123456789, 123456789],
                "imo": ["9123456", "9123456"],
                "vessel_name": ["Ship A", "Ship A"],
                "dep_locode": ["USNYC", "USNYC"],
                "arr_locode": ["GBLON", "ZASSB"],
                "dep_lat": [40.7128, 40.7128],
                "dep_lon": [-74.0060, -74.0060],
                "arr_lat": [51.5074, -29.8587],
                "arr_lon": [-0.1278, 31.0218],
                "arr_time": ["2025-01-01 12:00:00", "2025-01-02 14:00:00"],
                "dep_time": ["2025-01-01 10:00:00", "2025-01-02 12:00:00"],
            }
        )
        mock_con.execute.return_value.pl.return_value = mock_df

        # Create temporary directories
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        silver_dir = tmp_path / "silver"
        silver_dir.mkdir()

        # Create custom settings with temp paths
        custom_paths = PathConfig(
            data_root=tmp_path,
            bronze=tmp_path / "bronze",
            silver=silver_dir,
            gold=gold_dir,
            reference=tmp_path / "reference",
        )
        settings = AppSettings(paths=custom_paths)

        with (
            patch("src.transform.GOLD_DIR", gold_dir),
            patch("src.transform.SILVER_DIR", silver_dir),
        ):
            stitch_voyages(2025, 1, settings=settings)

            # Verify database was called
            mock_con.execute.assert_called_once()

            # Check that output file was created
            output_file = gold_dir / "voyages.parquet"
            assert output_file.exists()

    @patch("src.transform.duckdb.connect")
    @patch("src.config.GOLD_DIR", Path("/tmp/test_gold"))
    @patch("src.config.SILVER_DIR", Path("/tmp/test_silver"))
    def test_stitch_voyages_no_data(self, mock_duckdb_connect, tmp_path):
        """Test handling when no voyages are found."""
        mock_con = MagicMock()
        mock_duckdb_connect.return_value.__enter__.return_value = mock_con

        # Return empty DataFrame
        mock_con.execute.return_value.pl.return_value = pl.DataFrame()

        # Create custom settings with temp paths
        custom_paths = PathConfig(
            data_root=tmp_path,
            bronze=tmp_path / "bronze",
            silver=tmp_path / "silver",
            gold=tmp_path / "gold",
            reference=tmp_path / "reference",
        )
        settings = AppSettings(paths=custom_paths)

        stitch_voyages(2025, 1, settings=settings)

        # Should not create output file when no data
        # (This test mainly checks that no exception is raised)

    @patch("src.config.GOLD_DIR", Path("/tmp/test_gold"))
    @patch("src.config.SILVER_DIR", Path("/tmp/test_silver"))
    @patch("src.transform.duckdb.connect")
    def test_stitch_voyages_database_error(self, mock_duckdb_connect, tmp_path):
        """Test handling of database errors."""
        mock_duckdb_connect.return_value.__enter__.side_effect = Exception("DB error")

        # Create custom settings with temp paths
        custom_paths = PathConfig(
            data_root=tmp_path,
            bronze=tmp_path / "bronze",
            silver=tmp_path / "silver",
            gold=tmp_path / "gold",
            reference=tmp_path / "reference",
        )
        settings = AppSettings(paths=custom_paths)

        with pytest.raises(Exception):
            stitch_voyages(2025, 1, settings=settings)
