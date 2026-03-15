import streamlit as st
import duckdb
import pydeck as pdk

st.set_page_config(page_title="AIS Intelligence", layout="wide")

@st.cache_resource
def get_con():
    return duckdb.connect(database=':memory:')

def main():
    st.title("⚓ Global Maritime Intelligence")
    con = get_con()
    
    try:
        con.execute("CREATE OR REPLACE VIEW voyages AS SELECT * FROM read_parquet('/data/gold/*.parquet')")
        df_gold = con.execute("SELECT * FROM voyages").df()
    except Exception:
        st.info("No Gold data found. Please run the ETL pipeline first.")
        return

    st.subheader("🏙️ Global Shipping Lanes (3D)")
    
    layer = pdk.Layer(
        "ArcLayer",
        df_gold.dropna(subset=['distance_nm']),
        get_source_position=["lon", "lat"],
        get_target_position=["next_lon", "next_lat"],
        get_source_color=[0, 128, 255, 100],
        get_target_color=[255, 0, 128, 100],
        get_width=2,
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.5, pitch=45)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10"))

if __name__ == "__main__":
    main()
