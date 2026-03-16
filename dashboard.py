import os

import duckdb
import pydeck as pdk
import streamlit as st

from src.config import GOLD_DIR, get_logger

# Initialize professional logging for the UI
logger = get_logger(__name__)


def init_duckdb():
    """
    Initializes an in-memory DuckDB connection for high-speed dashboard queries.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the database.
    """
    return duckdb.connect(database=":memory:")


def load_data(con):
    """
    Creates a virtual view in DuckDB over the Gold Parquet files in the Modal Volume.

    Args:
        con (duckdb.DuckDBPyConnection): The active database connection.
    """
    gold_pattern = os.path.join(GOLD_DIR, "*.parquet")
    try:
        # We use a VIEW so DuckDB only reads the necessary columns during queries
        con.execute(
            f"CREATE OR REPLACE VIEW voyages AS SELECT * FROM read_parquet('{gold_pattern}')"
        )
        logger.info("Gold layer view successfully created in DuckDB.")
    except Exception as e:
        logger.error(f"Failed to load Gold data: {e}")
        st.error("No voyage data found. Please run the ETL pipeline first.")


def main():
    """
    Main Streamlit application entry point.
    Configures the 3D Pydeck environment and interactive filters.
    """
    st.set_page_config(page_title="AIS Voyage Engine", layout="wide", page_icon="⚓")

    st.title("⚓ Global Maritime Intelligence Dashboard")
    st.sidebar.header("Filters")

    con = init_duckdb()
    load_data(con)

    # --- Sidebar Controls ---
    min_dist = st.sidebar.slider("Minimum Distance (nm)", 0, 5000, 100)
    # vessel_type = st.sidebar.multiselect(
    #    "Vessel Type", ["Cargo", "Tanker", "Fishing", "Passenger"], default=["Cargo", "Tanker"]
    # )

    # --- Data Querying ---
    query = f"""
        SELECT * FROM voyages
        WHERE trip_distance_nm >= {min_dist}
    """
    # Execute query and convert to Pandas for Pydeck
    df = con.execute(query).df()

    # --- 3D Visualization (Pydeck) ---
    st.subheader(f"Showing {len(df)} Active Voyages")

    #
    arc_layer = pdk.Layer(
        "ArcLayer",
        df,
        get_source_position=["lon_departure", "lat_departure"],
        get_target_position=["lon_arrival", "lat_arrival"],
        get_source_color=[0, 255, 128, 80],  # Teal
        get_target_color=[255, 0, 128, 80],  # Pink
        get_width="1 + trip_distance_nm / 500",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(latitude=20.0, longitude=0.0, zoom=1.5, pitch=45, bearing=0)

    r = pdk.Deck(
        layers=[arc_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={
            "html": "<b>Ship:</b> {ship_name}<br/>"
            "<b>From:</b> {locode_departure}<br/>"
            "<b>To:</b> {locode_arrival}<br/>"
            "<b>Distance:</b> {trip_distance_nm} nm",
            "style": {"color": "white"},
        },
    )

    st.pydeck_chart(r)

    # --- Stats Section ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Voyages", len(df))
    with col2:
        avg_dist = df["trip_distance_nm"].mean() if not df.empty else 0
        st.metric("Avg Distance", f"{avg_dist:.1f} nm")
    with col3:
        unique_ships = df["imo"].nunique() if not df.empty else 0
        st.metric("Unique Vessels", unique_ships)


if __name__ == "__main__":
    main()
