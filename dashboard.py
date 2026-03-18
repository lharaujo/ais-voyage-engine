import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import pydeck as pdk
import streamlit as st


# Try to find the latest voyages file
def find_latest_voyages_file() -> Optional[str]:
    """Find the most recent voyages parquet file."""
    gold_dir = Path("data/gold")
    if not gold_dir.exists():
        return None

    voyages_files = list(gold_dir.glob("voyages*.parquet"))
    if not voyages_files:
        return None

    # Return the most recently modified file
    latest_file = max(voyages_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


DATA_SOURCE = find_latest_voyages_file() or "data/gold/voyages.parquet"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load voyage data from parquet file."""
    try:
        if not DATA_SOURCE or not os.path.exists(DATA_SOURCE):
            st.warning(f"⚠️ Data file not found: {DATA_SOURCE}")
            return pd.DataFrame()

        with duckdb.connect(database=":memory:") as con:
            query = f"SELECT * FROM read_parquet('{DATA_SOURCE}')"
            df = con.execute(query).df()
            return df
    except Exception as e:
        st.error(f"❌ Error loading data from {DATA_SOURCE}: {e}")
        return pd.DataFrame()


def render_vessel_view(data: pd.DataFrame, search: str, minimum_dist: int) -> None:
    """Render the vessel tracking visualization."""
    # Filter by minimum distance
    filtered_df = data[data["trip_distance_nm"] >= minimum_dist].copy()

    # Filter by search query if provided
    if search:
        search_upper = search.upper()
        is_ship = filtered_df["imo"].astype(str).str.upper().str.contains(
            search_upper
        ) | filtered_df["vessel_name"].astype(str).str.upper().str.contains(search_upper)
        filtered_df = filtered_df[is_ship]

    # Display metrics and info
    if search and not filtered_df.empty:
        ship_name = filtered_df["vessel_name"].iloc[0]
        imo_num = filtered_df["imo"].iloc[0]
        st.success(f"📍 Tracking: **{ship_name}** (IMO: {imo_num})")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Voyages this Month", len(filtered_df))
        with col2:
            total_dist = filtered_df["trip_distance_nm"].sum()
            st.metric("Total Distance (nm)", f"{total_dist:,.0f}")
        with col3:
            avg_speed = filtered_df["avg_speed_kts"].mean()
            st.metric("Avg Speed (kts)", f"{avg_speed:.1f}")
    elif search:
        st.error(f"🚫 No voyages found for '{search}' with current filters.")
    else:
        st.info("🌍 Displaying global traffic. Use the sidebar to track a specific ship.")

    # Create visualization
    if not filtered_df.empty:
        line_color = [255, 255, 0, 200] if search else [0, 255, 128, 140]
        line_width = 5 if search else 2

        # Ensure path column exists and is valid
        if "path" in filtered_df.columns:
            # Filter out rows with invalid paths
            valid_paths = filtered_df["path"].apply(
                lambda x: isinstance(x, list)
                and len(x) > 0
                and all(isinstance(pt, list) and len(pt) == 2 for pt in x)
            )
            filtered_df = filtered_df[valid_paths]

        if not filtered_df.empty:
            path_layer = pdk.Layer(
                "PathLayer",
                filtered_df,
                get_path="path",
                get_color=line_color,
                get_width=line_width,
                width_min_pixels=2,
                pickable=True,
            )

            deck = pdk.Deck(
                layers=[path_layer],
                initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.5),
                map_style="mapbox://styles/mapbox/navigation-night-v1",
                tooltip={"text": "Vessel: {vessel_name}\nFrom: {dep_locode} to {arr_locode}"},
            )

            st.pydeck_chart(deck)
        else:
            st.warning("⚠️ No valid voyage paths to display.")
    else:
        st.info("ℹ️ No data to display with current filters.")


def main() -> None:
    """Main dashboard application."""
    st.set_page_config(page_title="AIS Fleet Tracker", layout="wide", page_icon="⚓")

    df = load_data()
    if df.empty:
        st.warning("⚠️ No voyage data found. Please run the pipeline first.")
        return

    st.sidebar.header("🚢 Fleet Search")

    search_query = (
        st.sidebar.text_input(
            "Enter IMO or Vessel Name",
            placeholder="e.g. 9444728",
            help="Search for a specific ship to see its unique voyage history.",
        )
        .strip()
        .upper()
    )

    min_dist = st.sidebar.slider("Min Voyage Distance (nm)", 0, 5000, 50)

    render_vessel_view(df, search_query, min_dist)


if __name__ == "__main__":
    main()
