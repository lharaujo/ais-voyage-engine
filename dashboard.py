import glob

import duckdb
import pandas as pd
import pydeck as pdk
import streamlit as st

# Removed unused get_logger to satisfy F401
try:
    from src.config import GOLD_DIR
except ImportError:
    GOLD_DIR = "/data/gold"


def init_duckdb():
    return duckdb.connect(database=":memory:")


def load_data():
    con = init_duckdb()
    gold_path = f"{GOLD_DIR}/*.parquet"

    if not glob.glob(gold_path):
        return pd.DataFrame()

    try:
        con.execute(f"CREATE OR REPLACE VIEW voyages AS SELECT * FROM read_parquet('{gold_path}')")
        return con.execute("SELECT * FROM voyages").df()
    except Exception as e:  # Fixed E722
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    st.set_page_config(page_title="AIS Voyage Intelligence", layout="wide", page_icon="⚓")

    raw_df = load_data()

    if raw_df.empty:
        st.warning("⚠️ No data available. Please ensure the pipeline has run.")
        return

    # --- Sidebar ---
    min_dist = st.sidebar.slider("Min Distance (nm)", 0, 5000, 100)
    df = raw_df[raw_df["trip_distance_nm"] >= min_dist]

    # --- Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Voyages", f"{len(df):,}")

    # Fixed F841: Use calculation directly in metric to avoid unused variable
    avg_speed_val = df["avg_speed_kts"].mean() if not df.empty else 0.0
    m2.metric("Avg Speed", f"{avg_speed_val:.1f} kts")

    m3.metric("Max Distance", f"{df['trip_distance_nm'].max():,.0f} nm")
    m4.metric("Unique Vessels", f"{df['mmsi'].nunique():,}")

    # --- Map ---
    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.5, pitch=45)
    arc_layer = pdk.Layer(
        "ArcLayer",
        df,
        get_source_position=["lon_departure", "lat_departure"],
        get_target_position=["lon_arrival", "lat_arrival"],
        get_source_color=[0, 255, 128, 120],
        get_target_color=[255, 0, 128, 120],
        pickable=True,
    )

    st.pydeck_chart(pdk.Deck(layers=[arc_layer], initial_view_state=view_state))


if __name__ == "__main__":
    main()
