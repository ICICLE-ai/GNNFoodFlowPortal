import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
import matplotlib.pyplot as plt
import math

# External links (configure as needed)
GITHUB_URL = "https://github.com/ICICLE-ai/GNNFoodFlowPortal"
DOCS_URL = "https://doi.org/10.1145/3748636.3764168"
ICICLE_URL = "https://icicle.osu.edu/"

st.set_page_config(
    page_title="FAF Food Flows Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* Global Styles */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  * {
    font-family: 'Inter', sans-serif;
  }

  /* Hide default elements */
  #MainMenu, footer, header {visibility:hidden;}
  /* Re-enable sidebar toggle */
  [data-testid="collapsedControl"] {display: block !important;}

  /* Main container styling */
  .block-container {
    padding: 0 1rem;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    min-height: 100vh;
  }

  /* Ensure the entire page has consistent background */
  .main .block-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    min-height: 100vh;
  }

  /* Apply background to the entire app */
  .stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  }

  /* Ensure all content areas have consistent background */
  .stApp > div {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  }

  /* Sidebar styling */
  .css-1d391kg {
    background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    border-right: 1px solid #475569;
  }

  .css-1d391kg .element-container {
    margin-bottom: 1.5rem;
  }

  /* Button styling */
  button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    border-radius: 12px;
    color: white;
    font-weight: 600;
    border: none;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
  }

  button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
  }

  /* Map toggle button styling */
  .map-toggle-btn {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 8px;
    color: white;
    font-weight: 600;
    border: none;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    margin-bottom: 1rem;
  }

  .map-toggle-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
  }

  /* Heading styles */
  h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    text-align: center;
    letter-spacing: -0.02em;
  }

  h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #374151;
    margin-top: 0;
    margin-bottom: 1rem;
  }

  h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #4b5563;
    margin-bottom: 0.75rem;
  }

  /* Welcome screen styling */
  .welcome-container {
    text-align: center;
    padding: 5rem 3rem;
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 24px;
    margin: 2rem 0;
    box-shadow: 0 20px 60px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
  }

  .welcome-title {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    text-shadow: none;
    letter-spacing: -0.03em;
    line-height: 1.1;
    max-width: 24ch; /* keep to ~2 lines */
    margin-left: auto;
    margin-right: auto;
  }

  .welcome-subtitle {
    font-size: 1.3rem;
    color: #6b7280;
    margin-bottom: 0.75rem;
    line-height: 1.6;
    font-weight: 500;
    max-width: 60ch;
    margin-left: auto;
    margin-right: auto;
  }

  .welcome-why {
    font-size: 1rem;
    color: #374151;
    margin-bottom: 2rem;
    line-height: 1.6;
    font-weight: 500;
  }

  .start-button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    border: none;
    padding: 1.25rem 3.5rem;
    font-size: 1.4rem;
    font-weight: 700;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.4s ease;
    box-shadow: 0 8px 30px rgba(5, 150, 105, 0.3);
    letter-spacing: 0.02em;
  }

  .start-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(5, 150, 105, 0.4);
  }

  /* Feature list styling */
  .feature-list {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 3rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.06);
  }

  .feature-list h3 {
    color: #1f2937;
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .feature-list ul {
    list-style: none;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
  }

  .feature-list li {
    background: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    border: 1px solid #e5e7eb;
    transition: all 0.3s ease;
  }

  .feature-list li:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
  }

  /* Card styling */
  .metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid rgba(255,255,255,0.8);
    margin-bottom: 1rem;
  }

  /* Expander styling */
  .stExpanderHeader {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
    color: #0c4a6e !important;
    border-radius: 12px !important;
    border: 1px solid #bae6fd !important;
    font-weight: 600 !important;
  }

  /* Table styling */
  .stTable {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  }

  .stTable th {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    color: #1f2937 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #e5e7eb !important;
  }

  .stTable td {
    background: white !important;
    color: #374151 !important;
    border-bottom: 1px solid #f3f4f6 !important;
  }

  .stTable tr:hover td {
    background: #f8fafc !important;
  }

  /* Metric styling */
  [data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
  }

  [data-testid="stMetricValue"] {
    color: #059669 !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
  }

  /* Selectbox styling */
  .stSelectbox > div > div {
    border-radius: 8px !important;
    border: 2px solid #e5e7eb !important;
  }

  .stSelectbox > div > div:hover {
    border-color: #3b82f6 !important;
  }

  /* Slider styling */
  .stSlider > div > div > div > div {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
  }

  .stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    color: #6b7280;
    font-weight: 500;
    padding: 0.5rem 1rem;
  }

  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ffffff 0%, #e0f2fe 100%);
    color: #1f2937;
    border-color: #3b82f6;
    box-shadow: inset 0 -3px 0 #1d4ed8; /* subtle active underline */
  }

  /* Download button styling */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 8px;
    color: white;
    font-weight: 600;
    border: none;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
  }

  .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
  }

  /* Warning styling */
  .stAlert {
    border-radius: 12px;
    border: 1px solid #fef3c7;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
  }

  /* Code block styling */
  code {
    background: #f1f5f9;
    color: #0f172a;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.875rem;
    border: 1px solid #e2e8f0;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .welcome-title {
      font-size: 2.5rem;
    }

    .welcome-subtitle {
      font-size: 1.1rem;
    }

    .feature-list ul {
      grid-template-columns: 1fr;
    }
  }
</style>
""", unsafe_allow_html=True)

# Global top-right links bar
st.markdown(
    f"""
    <div style="position: fixed; top: 10px; right: 14px; z-index: 9999; display: flex; gap: 8px;">
        <a href="{GITHUB_URL}" target="_blank" style="text-decoration:none;">
            <span style="display:inline-block; padding:8px 12px; border-radius:10px; background:linear-gradient(135deg,#111827 0%, #1f2937 100%); color:#f9fafb; font-weight:600; box-shadow:0 4px 12px rgba(0,0,0,0.15); border:1px solid #374151;">🔗 GitHub</span>
        </a>
        <a href="{DOCS_URL}" target="_blank" style="text-decoration:none;">
            <span style="display:inline-block; padding:8px 12px; border-radius:10px; background:linear-gradient(135deg,#0ea5e9 0%, #3b82f6 100%); color:#ffffff; font-weight:600; box-shadow:0 4px 12px rgba(59,130,246,0.25); border:1px solid rgba(255,255,255,0.2);">📘 Paper</span>
        </a>
        <a href="{ICICLE_URL}" target="_blank" style="text-decoration:none;">
            <span style="display:inline-block; padding:8px 12px; border-radius:10px; background:linear-gradient(135deg,#f87171 0%, #dc2626 100%); color:#ffffff; font-weight:600; box-shadow:0 4px 12px rgba(239,68,68,0.18); border:1px solid rgba(255,255,255,0.2);"> ICICLE</span>
        </a>

    </div>
    """,
    unsafe_allow_html=True,
)


# Initialize session state for app state
if 'app_started' not in st.session_state:
    st.session_state['app_started'] = False

# Map size state (removed toggle functionality)

FAF_FILES = {
    "01 - Live Animals/Fish": "predicted_sctg_1.csv",
    "02 - Cereal Grains": "predicted_sctg_2.csv",
    "03 - Other Ag products.": "predicted_sctg_3.csv",
    "04 - Animal Feed": "predicted_sctg_4.csv",
    "05 - Meat/Seafood": "predicted_sctg_5.csv",
    "06 - Milled Grain Prods.": "predicted_sctg_6.csv",
    "07 - Other Foodstuffs": "predicted_sctg_7.csv"
}

@st.cache_data
def load_sctg_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(f"./cleaned_data/{file_name}")
    df['origin'] = df['origin'].astype(str).str.zfill(5)
    df['dest']   = df['dest'].astype(str).str.zfill(5)
    df = df.query("exist_prob > 0.5 and predicted_value_original > 0")
    # Filter out self-loops (origin == destination)
    df = df.query("origin != dest")
    return df

@st.cache_data
def load_county_metadata() -> pd.DataFrame:
    meta = pd.read_csv("cleaned_data/state_and_county_fips_master.csv")
    meta['fips'] = meta['fips'].astype(str).str.zfill(5)
    meta = meta.rename(columns={"fips": "FIPS", "name": "County", "state": "State"})
    return meta[['FIPS', 'County', 'State']]

@st.cache_data
def load_county_boundaries() -> dict:
    gdf = gpd.read_file("data/shapefiles/cb_2017_us_county_500k/cb_2017_us_county_500k.shp")
    gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)
    gdf = gdf.rename(columns={'GEOID': 'FIPS'}).merge(load_county_metadata(), on="FIPS", how="left")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)
    return json.loads(gdf.to_json())

def convert_sctg_to_trip(sctg_df, meta_df):
    # Create origin and destination labels with county names
    origin_labels = []
    dest_labels = []

    for orig, dest in zip(sctg_df["origin"], sctg_df["dest"]):
        # Get origin county info
        orig_info = meta_df[meta_df.FIPS == orig]
        if len(orig_info) > 0:
            orig_label = f"{orig_info['County'].iloc[0]}, {orig_info['State'].iloc[0]}"
        else:
            orig_label = f"Unknown ({orig})"

        # Get destination county info
        dest_info = meta_df[meta_df.FIPS == dest]
        if len(dest_info) > 0:
            dest_label = f"{dest_info['County'].iloc[0]}, {dest_info['State'].iloc[0]}"
        else:
            dest_label = f"Unknown ({dest})"

        origin_labels.append(orig_label)
        dest_labels.append(dest_label)

    return pd.DataFrame({
        "coordinates": [
            [[ox, oy], [dx, dy]]
            for ox, oy, dx, dy in zip(
                sctg_df["origin_x"], sctg_df["origin_y"],
                sctg_df["dest_x"], sctg_df["dest_y"]
            )
        ],
        "timestamps": [[0, 2]] * len(sctg_df),
        "orig_dms": origin_labels,
        "dest_dms": dest_labels,
        "origin": sctg_df["origin"].tolist(),
        "dest": sctg_df["dest"].tolist(),
        "predicted_value_original": sctg_df["predicted_value_original"].tolist()
    })

# Load data
META  = load_county_metadata()
BNDJS = load_county_boundaries()

# Welcome Screen or Main App
if not st.session_state['app_started']:
    # 1:6:1 layout
    # Place the logo and title in the center column (the "8" in 1:8:1), side by side, same height, without using the two "1" columns
    left, center, right = st.columns([1, 8, 1])

    with center:
        st.markdown(
            """
            <div class="welcome-container" style="display: flex; align-items: center; height: 100%;">
                <div style="flex: 0 0 30%;">
                    <img src="https://raw.githubusercontent.com/ICICLE-ai/GNNFoodFlowPortal/main/portal/image/ICICLE.png"
                        style="width:100%; height:auto; display:block;">
                </div>
                <div style="flex: 1; margin-left: 2rem;">
                    <div class="welcome-title" style="font-size:5rem; font-weight:700; margin-bottom:0.5rem;">
                        🌾 The Food Story of Franklin County, Ohio
                    </div>
                    <div class="welcome-subtitle" style="font-size:3rem; color:#334155; margin-bottom:0.25rem;">
                        Where Franklin County's Food Comes From.
                    </div>
                    <div class="welcome-why" style="font-size:2rem; color:#64748b;">
                        Why it matters: Disruptions hundreds of miles away can empty local shelves.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Franklin County locator mini‑map

        # Create tabs for different food categories
        tab1, tab2, tab3 = st.tabs(["🥩 Meat, Poultry, Fish & Seafood", "🌽 Agricultural Products", "🍽️ Other Foodstuffs"])

        with tab1:
            # === Header: SCTG05 ===
            st.markdown("""
            <div style="margin:1rem 0; padding:1.5rem; border-radius:16px;
                        background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);
                        box-shadow:0 6px 20px rgba(0,0,0,0.06);">
                <h3>🥩 Meat, Poultry, Fish & Seafood (SCTG05)</h3>
            </div>
            """, unsafe_allow_html=True)

            # === Two columns ===
            col_text, col_img = st.columns([2, 1])

            with col_text:
                st.markdown("""
                <div style="padding: 0 1rem 0 0;">
                    <p style="font-size:1rem; color:#374151; line-height:1.6; margin:0;">
                    Franklin’s supply of meat, poultry, fish, and seafood does not only come from nearby farms.
                    It depends heavily on processing plants and cold-chain routes from Chicago, Detroit, Cincinnati, Philadelphia, New York, and other metropolitan hubs.
                    <br><br>
                    These protein staples travel longer supply chains, requiring refrigeration every step of the way.
                    This makes them especially vulnerable: disruptions at distant plants or highway closures can ripple quickly into Columbus grocery shelves.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_img:
                st.image(
                    "image/SCTG05.jpg",
                    use_container_width=True
                )
                st.markdown(
                    """
                    <div style="text-align:center; font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                        Meat, Poultry, Fish & Seafood (SCTG05)
                        <br>
                        <a href="https://www.supermarketnews.com/meat/value-added-meats-now-packed-with-even-more-value"
                            target="_blank"
                            style="color:#2563eb;text-decoration:underline;">
                            Img Source
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # === Data (Top 10 origins) ===
            franklin_lat, franklin_lon = 39.9612, -82.9988  # Franklin County, OH (Columbus)

            sctg05_data = pd.DataFrame([
                {"origin": "Cook, IL (Chicago)",        "lat": 41.8781, "lon": -87.6298, "kilotons": 5186.5215, "pct": 1.989805},
                {"origin": "Marion, IN (Indianapolis)", "lat": 39.7684, "lon": -86.1581, "kilotons": 1973.1515, "pct": 0.756998},
                {"origin": "Wayne, MI (Detroit)",       "lat": 42.3314, "lon": -83.0458, "kilotons": 1934.1696, "pct": 0.742043},
                {"origin": "Hamilton, OH (Cincinnati)", "lat": 39.1031, "lon": -84.5120, "kilotons": 1842.3495, "pct": 0.706816},
                {"origin": "Oakland, MI",               "lat": 42.5917, "lon": -83.3362, "kilotons": 1789.3154, "pct": 0.686470},
                {"origin": "Cuyahoga, OH (Cleveland)",  "lat": 41.4993, "lon": -81.6944, "kilotons": 1649.1882, "pct": 0.632710},
                {"origin": "Kings, NY (Brooklyn)",      "lat": 40.6782, "lon": -73.9442, "kilotons": 1624.1476, "pct": 0.623103},
                {"origin": "Philadelphia, PA",          "lat": 39.9526, "lon": -75.1652, "kilotons": 1576.2966, "pct": 0.604745},
                {"origin": "Allegheny, PA (Pittsburgh)","lat": 40.4406, "lon": -79.9959, "kilotons": 1545.4667, "pct": 0.592917},
                {"origin": "Macomb, MI",                "lat": 42.6730, "lon": -82.9199, "kilotons": 1503.3323, "pct": 0.576752},
            ])

            # === “At a glance” + bar chart ===
            st.markdown("""
            <div style="margin-top:0.5rem; margin-bottom:0.5rem; color:#6b7280; font-weight:600;">At a glance</div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Top Source", sctg05_data.iloc[0]["origin"], f"{sctg05_data.iloc[0]['pct']:.2f}%")
                st.metric("Top 10 Share", f"{sctg05_data['pct'].sum():.1f}%", "of total inflow")
            with col2:
                sorted_data = sctg05_data.sort_values("kilotons", ascending=False)

                # Add rank to origin name
                sorted_data["label"] = (sorted_data.index + 1).astype(str).str.zfill(len(str(len(sorted_data)))) + " - " + sorted_data["origin"]

                st.bar_chart(sorted_data.set_index("label")["kilotons"])


            # === Mini map (PyDeck) ===
            arc_data = []
            for _, row in sctg05_data.iterrows():
                arc_data.append({
                    "from": [row["lon"], row["lat"]],
                    "to": [franklin_lon, franklin_lat],
                    "origin": row["origin"],
                    "kilotons": row["kilotons"],
                    "pct": row["pct"]
                })
            arc_df = pd.DataFrame(arc_data)

            arc_layer = pdk.Layer(
                "ArcLayer",
                data=arc_df,
                get_source_position="from",
                get_target_position="to",
                get_source_color=[249,115,22,180],  # orange
                get_target_color=[59,130,246,180],  # blue
                get_width="kilotons",
                width_scale=0.00005,
                width_min_pixels=2,
                width_max_pixels=10,
                pickable=True,
                auto_highlight=True,
            )

            franklin_row = pd.DataFrame([{
                "origin": "Franklin County, OH (Columbus)",
                "lat": franklin_lat,
                "lon": franklin_lon,
                "kilotons": "N/A",
                "pct": "N/A"
            }])
            scatter_data = pd.concat([sctg05_data, franklin_row], ignore_index=True)
            scatter_data["color"] = scatter_data["origin"].apply(
                lambda x: [59,130,246,220] if x.startswith("Franklin County") else [249,115,22,200]
            )

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=scatter_data,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=60000,
                pickable=True,
            )

            view = pdk.ViewState(latitude=40, longitude=-83, zoom=5, pitch=30)
            st.pydeck_chart(pdk.Deck(
                layers=[arc_layer, scatter_layer],
                initial_view_state=view,
                tooltip={"html": "<b>{origin}</b><br/>Estimated kilotons: {kilotons}<br/>Estimated Share: {pct}%"}
            ), height=600)

            st.caption("Flows to Franklin County (blue) from top 10 origins (orange). Hover to see details.")

        with tab2:
            # Create a container with the background styling
            st.markdown("""
            <div style="margin:1rem 0; padding:1.5rem; border-radius:16px;
                        background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);
                        box-shadow:0 6px 20px rgba(0,0,0,0.06);">
                <h3>🌽 Agricultural Products (SCTG03)</h3>
            </div>
            """, unsafe_allow_html=True)

            # Create columns for text and image
            col_text, col_img = st.columns([2, 1])

            with col_text:
                st.markdown("""
                <div style="padding: 0 1rem 0 0;">
                    <p style="font-size:1rem; color:#374151; line-height:1.6; margin:0;">
                    Behind every loaf of bread or box of cereal in Franklin County is a web stretching to Chicago, Houston, Los Angeles, and Seattle.
                    These cities are the nation's food arteries, pushing vast flows of grain and other crops toward Ohio.
                    <br><br>
                    It looks strong, but the dependence is sharp: when floods close the rail lines in Chicago, or hurricanes strike Houston, Franklin's steady stream of grain falters.
                    This isn't just a question of one meal—it's a reminder that systemic risks ripple through the heart of America's food supply.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_img:
                st.image(
                    "image/SCTG03.jpg",
                    use_container_width=True
                )
                st.markdown(
                    """
                    <div style="text-align:center; font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                        Agricultural Products (SCTG03)
                        <br>
                        <a href="https://www.agritechtomorrow.com/article/2024/08/e-commerce-of-agricultural-products-market-pioneering-the-digital-transformation-of-agriculture/15785"
                            target="_blank"
                            style="color:#2563eb;text-decoration:underline; font-size: 0.8rem;">
                            Img Source
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # SCTG03 data and visualization
            sctg03_data = pd.DataFrame([
                {"origin": "Cook, IL (Chicago)", "lat": 41.8781, "lon": -87.6298, "kilotons": 105572.5, "pct": 6.40},
                {"origin": "Montgomery, TX (Houston Metro)", "lat": 30.3213, "lon": -95.4778, "kilotons": 104218.4, "pct": 6.31},
                {"origin": "Travis, TX (Austin)", "lat": 30.2672, "lon": -97.7431, "kilotons": 67072.6, "pct": 4.06},
                {"origin": "King, WA (Seattle)", "lat": 47.6062, "lon": -122.3321, "kilotons": 54311.0, "pct": 3.29},
                {"origin": "Los Angeles, CA", "lat": 34.0522, "lon": -118.2437, "kilotons": 52315.4, "pct": 3.17},
                {"origin": "Hidalgo, TX (Border)", "lat": 26.1004, "lon": -98.2636, "kilotons": 43075.0, "pct": 2.61},
                {"origin": "Williamson, TX (Austin Metro)", "lat": 30.6400, "lon": -97.6800, "kilotons": 28010.2, "pct": 1.70},
                {"origin": "San Diego, CA", "lat": 32.7157, "lon": -117.1611, "kilotons": 26221.1, "pct": 1.59},
                {"origin": "Rutherford, TN (Nashville Metro)", "lat": 35.8456, "lon": -86.3903, "kilotons": 25528.0, "pct": 1.55},
                {"origin": "Chesterfield, VA (Richmond Metro)", "lat": 37.3778, "lon": -77.5040, "kilotons": 24948.2, "pct": 1.51},
            ])

            st.markdown("""
            <div style="margin-top:0.5rem; margin-bottom:0.5rem; color:#6b7280; font-weight:600;">At a glance</div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Top Source", sctg03_data.iloc[0]["origin"], f"{sctg03_data.iloc[0]['pct']:.2f}%")
                st.metric("Top 10 Share", f"{sctg03_data['pct'].sum():.1f}%", "of total inflow")
            with col2:
                sorted_data = sctg03_data.sort_values("kilotons", ascending=False)
                sorted_data["label"] = (sorted_data.index + 1).astype(str).str.zfill(len(str(len(sorted_data)))) + " - " + sorted_data["origin"]
                st.bar_chart(sorted_data.set_index("label")["kilotons"])

            # Mini map
            arc_data03 = []
            for _, row in sctg03_data.iterrows():
                arc_data03.append({
                    "from": [row["lon"], row["lat"]],
                    "to": [franklin_lon, franklin_lat],
                    "origin": row["origin"],
                    "kilotons": row["kilotons"],
                    "pct": row["pct"]
                })
            arc_df03 = pd.DataFrame(arc_data03)

            arc_layer03 = pdk.Layer(
                "ArcLayer",
                data=arc_df03,
                get_source_position="from",
                get_target_position="to",
                get_source_color=[249,115,22,180],
                get_target_color=[59,130,246,180],
                get_width="kilotons",
                width_scale=0.00001,
                width_min_pixels=2,
                width_max_pixels=12,
                pickable=True,
                auto_highlight=True,
            )

            franklin_point03 = pd.DataFrame([{
                "origin": "Franklin County, OH",
                "lat": franklin_lat, "lon": franklin_lon,
                "kilotons": "N/A", "pct": "N/A"
            }])
            scatter_data03 = pd.concat([sctg03_data, franklin_point03], ignore_index=True)
            scatter_data03['color'] = scatter_data03['origin'].apply(lambda x: [59,130,246,220] if x == "Franklin County, OH" else [249,115,22,200])

            scatter_layer03 = pdk.Layer(
                "ScatterplotLayer",
                data=scatter_data03,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=70000,
                pickable=True,
            )

            view03 = pdk.ViewState(latitude=39, longitude=-95, zoom=3, pitch=30)
            st.pydeck_chart(pdk.Deck(
                layers=[arc_layer03, scatter_layer03],
                initial_view_state=view03,
                tooltip={"html": "<b>{origin}</b><br/>Estimated Kilokilotons: {kilotons}<br/>Estimated Share: {pct}%"}
            ), height=600)
            st.caption("Flows to Franklin County (blue) from top 10 origins (orange). Hover over to see more details.")

        with tab3:
            # Create a container with the background styling
            st.markdown("""
            <div style="margin:1rem 0; padding:1.5rem; border-radius:16px;
                        background:linear-gradient(135deg,#fff7ed 0%,#ffedd5 100%);
                        box-shadow:0 6px 20px rgba(0,0,0,0.06);">
                <h3>🍽️ Other Foodstuffs (SCTG07)</h3>
            </div>
            """, unsafe_allow_html=True)

            # Create columns for text and image
            col_text, col_img = st.columns([2, 1])

            with col_text:
                st.markdown("""
                <div style="padding: 0 1rem 0 0;">
                    <p style="font-size:1rem; color:#374151; line-height:1.6; margin:0;">
                    Other foodstuffs (SCTG07) are the foods that make modern life delicious and convenient—think of the things you reach for every day that aren’t just raw ingredients. This category includes everything from milk and cream on your cereal to cheese on your pizza, along with your morning coffee. It also covers ice cream, jams, peanut butter, fruit juices, and a variety of oils and spreads.
                    The widest part of Franklin's diet comes not from Ohio at all, but from everywhere: Chicago, Phoenix, Los Angeles, New York, Houston, even Fort Lauderdale. These flows bring packaged foods, imports, and essentials that stock every aisle of the grocery store.
                    <br><br>
                    It feels invisible—until disaster hits.
                    Wildfires in California, storm surges in New York Harbor, hurricanes sweeping the Gulf: all can empty Franklin's shelves, thousands of miles away from the disaster itself.
                    Here, food security depends not only on farms, but on the resilience of ports and global trade itself.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_img:
                st.image(
                    "image/SCTG07.jpg",
                    use_container_width=True
                )
                st.markdown(
                    """
                    <div style="text-align:center; font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                        Other Foodstuffs (SCTG07)
                        <br>
                        <a href="https://www.nytimes.com/2017/02/13/well/eat/got-almond-milk-dairy-farms-protest-milk-label-on-nondairy-drinks.html"
                            target="_blank"
                            style="color:#2563eb;text-decoration:underline; font-size: 0.8rem;">
                            Img Source 1
                        </a>
                        <br>
                        <a href="https://aswathicherkkil.medium.com/marketing-ready-to-eat-products-17eb60e8e13b"
                            target="_blank"
                            style="color:#2563eb;text-decoration:underline; font-size: 0.8rem;">
                            Img Source 2
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # SCTG07 data and visualization
            sctg07_data = pd.DataFrame([
                {"origin": "Cook, IL (Chicago)", "lat": 41.8781, "lon": -87.6298, "kilotons": 228634.6, "pct": 7.05},
                {"origin": "Maricopa, AZ (Phoenix)", "lat": 33.4484, "lon": -112.0740, "kilotons": 119806.2, "pct": 3.69},
                {"origin": "Los Angeles, CA", "lat": 34.0522, "lon": -118.2437, "kilotons": 91113.4, "pct": 2.81},
                {"origin": "Kings, NY (Brooklyn)", "lat": 40.6782, "lon": -73.9442, "kilotons": 62833.8, "pct": 1.94},
                {"origin": "Harris, TX (Houston)", "lat": 29.7604, "lon": -95.3698, "kilotons": 62472.0, "pct": 1.93},
                {"origin": "Broward, FL (Fort Lauderdale)", "lat": 26.1901, "lon": -80.3659, "kilotons": 51277.5, "pct": 1.58},
                {"origin": "Queens, NY (New York City)", "lat": 40.7282, "lon": -73.7949, "kilotons": 46579.8, "pct": 1.44},
                {"origin": "San Bernardino, CA", "lat": 34.6214, "lon": -116.3225, "kilotons": 44528.8, "pct": 1.37},
                {"origin": "Clark, NV (Las Vegas)", "lat": 36.1699, "lon": -115.1398, "kilotons": 42667.0, "pct": 1.31},
                {"origin": "Tarrant, TX (Dallas–Fort Worth)", "lat": 32.7688, "lon": -97.3093, "kilotons": 42022.5, "pct": 1.30},
            ])

            st.markdown("""
            <div style="margin-top:0.5rem; margin-bottom:0.5rem; color:#6b7280; font-weight:600;">At a glance</div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Top Source", sctg07_data.iloc[0]["origin"], f"{sctg07_data.iloc[0]['pct']:.2f}%")
                st.metric("Top 10 Share", f"{sctg07_data['pct'].sum():.1f}%", "of total inflow")
            with col2:
                sorted_data = sctg07_data.sort_values("kilotons", ascending=False)
                sorted_data["label"] = (sorted_data.index + 1).astype(str).str.zfill(len(str(len(sorted_data)))) + " - " + sorted_data["origin"]
                st.bar_chart(sorted_data.set_index("label")["kilotons"])

            # Mini map
            arc_data07 = []
            for _, row in sctg07_data.iterrows():
                arc_data07.append({
                    "from": [row["lon"], row["lat"]],
                    "to": [franklin_lon, franklin_lat],
                    "origin": row["origin"],
                    "kilotons": row["kilotons"],
                    "pct": row["pct"]
                })
            arc_df07 = pd.DataFrame(arc_data07)

            arc_layer07 = pdk.Layer(
                "ArcLayer",
                data=arc_df07,
                get_source_position="from",
                get_target_position="to",
                get_source_color=[249,115,22,160],
                get_target_color=[59,130,246,200],
                get_width="kilotons",
                width_scale=0.000005,
                width_min_pixels=2,
                width_max_pixels=14,
                pickable=True,
                auto_highlight=True,
            )

            franklin_point07 = pd.DataFrame([{
                "origin": "Franklin County, OH",
                "lat": franklin_lat, "lon": franklin_lon,
                "kilotons": "N/A", "pct": "N/A"
            }])
            scatter_data07 = pd.concat([sctg07_data, franklin_point07], ignore_index=True)
            scatter_data07['color'] = scatter_data07['origin'].apply(lambda x: [59,130,246,220] if x == "Franklin County, OH" else [249,115,22,200])

            scatter_layer07 = pdk.Layer(
                "ScatterplotLayer",
                data=scatter_data07,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=80000,
                pickable=True,
            )

            view07 = pdk.ViewState(latitude=39, longitude=-96, zoom=3, pitch=30)
            st.pydeck_chart(pdk.Deck(
                layers=[arc_layer07, scatter_layer07],
                initial_view_state=view07,
                tooltip={"html": "<b>{origin}</b><br/>Estimated Kilotons: {kilotons}<br/>Estimated Share: {pct}%"}
            ), height=600)
            st.caption("Flows to Franklin County (blue) from top 10 origins (orange). Hover over to see more details.")

        # Call to action
        # Combine call-to-action and start button in a single centered block, with working Streamlit button

        with st.container():
            st.markdown("""
            <div style="margin:3rem 0; text-align:center; padding:2rem;
                        background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);
                        border-radius:20px; box-shadow:0 8px 32px rgba(0,0,0,0.08);">
                <h2 style="margin-bottom:1rem; color:#059669;">From Local Farms to Global Ports</h2>
                <p style="color:#4b5563; font-size:1.1rem; margin-bottom:2rem;">
                    Franklin County's food system is woven into regional, national, and global flows.
                    <br>
                    Now, let's explore the interactive map and see these flows come alive.
                </p>
            """, unsafe_allow_html=True)
            btn_clicked = st.button(
                "Explore the U.S. Map",
                key="start_button",
                use_container_width=True,
                type="primary"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if btn_clicked:
                st.session_state['app_started'] = True
                st.rerun()
else:
    # Main App Content
    # Sidebar
    with st.sidebar:
        st.image("./image/image17.png")

        # Back to Welcome button
        if st.button("Back to the Story", key="back_button"):
            st.session_state['app_started'] = False
            st.rerun()

        st.markdown("---")

        st.markdown("## Filter Options")
        if 'origin_lbl' not in st.session_state:
            st.session_state['origin_lbl'] = None
        if 'dest_lbl' not in st.session_state:
            st.session_state['dest_lbl'] = "All"

        cat = st.selectbox("Food Category (SCTG)", list(FAF_FILES))
        flows = load_sctg_data(FAF_FILES[cat])

        origins = sorted(flows['origin'].unique())
        origin_lbls = [f"{f} - {META.loc[META.FIPS==f,'County'].iat[0]} - {META.loc[META.FIPS==f,'State'].iat[0]}"
                       if f in META.FIPS.values else f for f in origins]
        origin_lbls = ["All"] + origin_lbls
        default_idx = origin_lbls.index("06037 - Los Angeles County - CA") if "06037 - Los Angeles County - CA" in origin_lbls else 0

        if st.session_state['origin_lbl'] is None:
            st.session_state['origin_lbl'] = origin_lbls[default_idx]
        origin_lbl = st.selectbox("Origin County", origin_lbls, index=origin_lbls.index(st.session_state['origin_lbl']) if st.session_state['origin_lbl'] in origin_lbls else default_idx, key="origin_select")
        st.session_state['origin_lbl'] = origin_lbl
        origin_fip = origin_lbl.split(" - ")[0] if origin_lbl != "All" else "All"

        if origin_lbl == "All":
            dests = sorted(flows['dest'].unique())
            filt = flows.copy()
        else:
            dests = sorted(flows.query("origin == @origin_fip")['dest'].unique())
            filt = flows.query("origin == @origin_fip")

        dest_lbls = [f"{d} - {META.loc[META.FIPS==d,'County'].iat[0]} - {META.loc[META.FIPS==d,'State'].iat[0]}"
                     if d in META.FIPS.values else d for d in dests]
        dest_options = ["All"] + dest_lbls
        if st.session_state['dest_lbl'] not in dest_options:
            st.session_state['dest_lbl'] = "All"
        dest_lbl = st.selectbox("Destination County", dest_options, index=dest_options.index(st.session_state['dest_lbl']), key="dest_select")
        st.session_state['dest_lbl'] = dest_lbl
        if dest_lbl != "All":
            dest_fip = dest_lbl.split(" - ")[0]
            filt = filt.query("dest == @dest_fip")

        if st.button("\u21c6 Swap Origin and Destination"):
            if st.session_state['dest_lbl'] == "All":
                st.session_state['origin_lbl'] = "All"
                st.session_state['dest_lbl'] = origin_lbl
            else:
                st.session_state['origin_lbl'], st.session_state['dest_lbl'] = st.session_state['dest_lbl'], st.session_state['origin_lbl']
            st.rerun()

        st.markdown("## Trip Summary")

        # Create metric cards with better styling
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #3b82f6; margin-bottom: 0.5rem;">
                        {len(filt):,}
                    </div>
                    <div style="font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                        Number of Trips
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #059669; margin-bottom: 0.5rem;">
                        {filt['predicted_value_original'].sum():,.0f}
                    </div>
                    <div style="font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                        Total Estimated Kilotons Shipped
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        top_n = st.slider("Top links to render", min_value=1, max_value=300, value=60, step=1)

        # Flow width scaling controls
        flow_scale_mode = st.selectbox(
            "Flow width scale",
            ["Linear (with cap)", "Logarithmic"],
            index=0,
            help="Choose how to scale arc widths: cap large flows in linear mode or compress range with logarithmic scaling."
        )
        if flow_scale_mode == "Linear (with cap)":
            suggested_cap = float(filt['predicted_value_original'].quantile(0.95)) if len(filt) > 0 else 1000.0
            max_flow_cap = st.number_input(
                "Max flow (Kilotons) cap",
                min_value=1.0,
                value=max(1.0, round(suggested_cap, 2)),
                step=1000.0,
                help="Any flow larger than this value will be drawn using this maximum width."
            )
        else:
            max_flow_cap = None

    # Main panel
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🌾 GNN Food Flow Portal</h1>
        <p style="color: #6b7280; font-size: 1.1rem; margin-top: -0.5rem;">
            Advanced Graph Neural Network Food Transportation Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs - REORDERED: Map first, then How to Use, then Download
    tab1, tab2, tab3 = st.tabs(["Interactive Map", "How to Use", "Download"])

    with tab1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    padding: 1.5rem;
                    border-radius: 16px;
                    border: 1px solid #bae6fd;
                    margin-bottom: 1.5rem;">
            <p style="margin: 0; color: #0369a1; font-size: 1.1rem; font-weight: 500;">
                Food Flows from <strong>{origin_lbl}</strong> to <strong>{dest_lbl}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)



        top_links = filt.nlargest(top_n, 'predicted_value_original')
        arc_df = convert_sctg_to_trip(top_links, META)

        # Compute width field based on selected scaling mode
        if flow_scale_mode == "Linear (with cap)":
            arc_df = arc_df.copy()
            arc_df['flow_width'] = arc_df['predicted_value_original'].clip(upper=max_flow_cap)
            width_scale_main = 0.00003
            width_scale_secondary = 0.000045
            width_scale_glow = 0.00006
        else:
            arc_df = arc_df.copy()
            arc_df['flow_width'] = arc_df['predicted_value_original'].apply(lambda v: math.log10(v + 1.0))
            # Larger scale factors because log values are small
            width_scale_main = 0.7
            width_scale_secondary = 1.0
            width_scale_glow = 1.2

        boundary_layer = pdk.Layer(
            "GeoJsonLayer", BNDJS,
            stroked=True, filled=True,
            get_fill_color=[240, 248, 255, 80],  # Light blue background
            get_line_color=[200, 200, 200, 120],  # Light gray borders
            line_width_min_pixels=1,
            pickable=False,
        )

        # Multiple ArcLayers to create gradient flow effect
        # Main arc layer with full gradient
        arc_layer_main = pdk.Layer(
            "ArcLayer", arc_df,
            get_source_position="coordinates[0]",
            get_target_position="coordinates[1]",
            get_source_color=[249, 115, 22, 180],  # Orange (origin)
            get_target_color=[59, 130, 246, 180],  # Blue (destination)
            get_width="flow_width",
            width_scale=width_scale_main,
            width_min_pixels=2,
            width_max_pixels=10,
            pickable=True,
            auto_highlight=True,
            get_tilt=15,
        )

        # Secondary arc layer for enhanced gradient effect
        arc_layer_secondary = pdk.Layer(
            "ArcLayer", arc_df,
            get_source_position="coordinates[0]",
            get_target_position="coordinates[1]",
            get_source_color=[249, 115, 22, 120],  # Orange with lower opacity
            get_target_color=[59, 130, 246, 120],  # Blue with lower opacity
            get_width="flow_width",
            width_scale=width_scale_secondary,
            width_min_pixels=1,
            width_max_pixels=6,
            pickable=False,
            get_tilt=15,
        )

        # Glow effect layer
        arc_layer_glow = pdk.Layer(
            "ArcLayer", arc_df,
            get_source_position="coordinates[0]",
            get_target_position="coordinates[1]",
            get_source_color=[249, 115, 22, 60],  # Orange glow
            get_target_color=[59, 130, 246, 60],  # Blue glow
            get_width="flow_width",
            width_scale=width_scale_glow,
            width_min_pixels=0,
            width_max_pixels=4,
            pickable=False,
            get_tilt=15,
        )

        view = pdk.ViewState(latitude=39.8283, longitude=-98.5795, zoom=3, pitch=45)

        # Set map height (increased default size)
        map_height = 600

        st.pydeck_chart(
            pdk.Deck(
                layers=[boundary_layer, arc_layer_glow, arc_layer_secondary, arc_layer_main],
                initial_view_state=view,
                tooltip={
                    "html": """
                    <div style="background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
                                color: white;
                                padding: 12px;
                                border-radius: 8px;
                                border: 1px solid #4b5563;
                                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                                font-family: 'Inter', sans-serif;">
                        <div style="margin-bottom: 8px;">
                            <span style="color: #10b981; font-weight: 600;">Origin:</span> {orig_dms}
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="color: #ef4444; font-weight: 600;">Destination:</span> {dest_dms}
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="color: #f59e0b; font-weight: 600;">Estimated Kilotons Shipped:</span> {predicted_value_original}
                        </div>
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #4b5563;">
                            <span style="color: #3b82f6; font-weight: 600;">FIPS: {origin} → {dest}</span>
                        </div>
                    </div>
                    """,
                    "style": {"backgroundColor": "transparent", "border": "none"}
                }
            ),
            height=map_height,
        )
        st.caption("Interactive U.S. flows: arcs transition from orange (origin) to blue (destination). Width scales with predicted kilotons. Hover for details.")

        origin = st.session_state['origin_lbl']
        dest = st.session_state['dest_lbl']
        if origin == "All" and dest != "All":
            expander_title = "Top 5 Origins by Estimated kilotons"
            group_col = 'origin'
        else:
            expander_title = "Top 5 Destinations by Estimated kilotons"
            group_col = 'dest'

        with st.expander(expander_title, expanded=True):
            table = (filt.groupby(group_col)['predicted_value_original']
                          .sum()
                          .sort_values(ascending=False)
                          .head(5)
                          .reset_index())

            table['County'] = table[group_col].map(lambda f: META.loc[META.FIPS==f, 'County'].iat[0] if f in META.FIPS.values else "")
            table['State'] = table[group_col].map(lambda f: META.loc[META.FIPS==f, 'State'].iat[0] if f in META.FIPS.values else "")

            st.table(table.rename(columns={group_col: 'FIPS', 'predicted_value_original': 'kilotons Shipped'}))

    with tab2:
        st.markdown("## How to Use the Food Flow Portal")

        st.markdown("""
        ### Welcome to the FAF Food Flows Dashboard!

        This interactive dashboard allows you to explore food transportation patterns across the United States using advanced Graph Neural Network predictions.

        ### Getting Started

        1. **Select Food Category**: Choose from 7 different food categories in the sidebar, refer to this [link](https://bhs.econ.census.gov/bhsphpext/brdsearch/scs_code.html) for detailed description of each category.
           - 01 - Live Animals/Fish
           - 02 - Cereal Grains
           - 03 -Other Agricultural Products
           - 04 - Animal Feed
           - 05 - Meat/Seafood
           - 06 - Milled Grain Products
           - 07 - Other Foodstuffs

        2. **Filter by Location**:
           - **Origin County**: Select where food shipments originate from
           - **Destination County**: Select where food shipments are going to
           - Use "All" to see all origins or destinations

        3. **Explore the Map**:
           - Interactive 3D map shows food flow trips between counties
           - Watch flows move dynamically from origin to destination
           - Hover over flows to see origin and destination details
           - Adjust the number of top links to display (1-300)

        4. **Analyze Data**:
           - View trip summary metrics in the sidebar
           - See top origins or destinations in the expandable table
           - Download filtered data for further analysis

        ### Interactive Features

        - **Swap Origin/Destination**: Quickly reverse your analysis
        - **Zoom & Pan**: Navigate the map to focus on specific regions
        - **Hover Effects**: Get detailed information about each flow
        - **Filter Combinations**: Mix and match categories and locations

        ### Understanding the Data

        - **Predicted Value**: Estimated kilotons of food shipped (based on GNN predictions)
        - **Existence Probability**: Confidence level of the predicted flow
        - **FIPS Codes**: Federal Information Processing Standards county identifiers

        ### Tips for Best Experience

        - Start with "All" origins to see major food flow patterns
        - Focus on specific counties for detailed local analysis
        - Use different food categories to understand supply chains
        - Combine filters to isolate specific trade relationships
        - Observe the arc flows to understand movement patterns
        """)


    with tab3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                    padding: 1.5rem;
                    border-radius: 16px;
                    border: 1px solid #bbf7d0;
                    margin-bottom: 2rem;">
            <h2 style="margin: 0 0 0.5rem 0; color: #166534;">Download Data</h2>
            <p style="margin: 0; color: #15803d; font-size: 1.1rem;">
                Export your filtered data in various formats for further analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📥 Download Options")

        # Create download butkilotons for different data formats
        col1, col3 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #e5e7eb;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
                        height: 100%;">
                <h4 style="margin: 0 0 0.75rem 0; color: #1f2937;">Current Filtered Data</h4>
                <p style="margin: 0 0 1.5rem 0; color: #6b7280; font-size: 0.9rem;">
                    Download the data being used for the map based on your selected filters.
                </p>
            """, unsafe_allow_html=True)

            # Add vertical space before the download button
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            if len(filt) > 0:
                csv_data = filt.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"food_flows_{cat.replace(' ', '_').lower()}_{origin_lbl.split(' - ')[0] if origin_lbl != 'All' else 'all'}_{dest_lbl.split(' - ')[0] if dest_lbl != 'All' else 'all'}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data available for current filters")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #e5e7eb;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
                        height: 100%;">
                <h4 style="margin: 0 0 0.75rem 0; color: #1f2937;">Summary Statistics</h4>
                <p style="margin: 0 0 1.5rem 0; color: #6b7280; font-size: 0.9rem;">
                    Download the flow information for the current food category.
                </p>
            """, unsafe_allow_html=True)

            # Add vertical space before the download button
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            if len(filt) > 0:
                summary_stats = filt.groupby(['origin', 'dest'])['predicted_value_original'].sum().reset_index()
                summary_stats = summary_stats.merge(META, left_on='origin', right_on='FIPS', how='left')
                summary_stats = summary_stats.merge(META, left_on='dest', right_on='FIPS', how='left', suffixes=('_origin', '_dest'))
                summary_stats = summary_stats[['origin', 'County_origin', 'State_origin', 'dest', 'County_dest', 'State_dest', 'predicted_value_original']]
                summary_stats.columns = ['Origin_FIPS', 'Origin_County', 'Origin_State', 'Dest_FIPS', 'Dest_County', 'Dest_State', 'kilotons_Shipped']

                summary_csv = summary_stats.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=summary_csv,
                    file_name=f"food_flows_summary_{cat.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data available for summary")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        # --- Data Description Card ---
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #fde68a;
                        margin-bottom: 1.5rem;">
                <h3 style="margin: 0 0 1rem 0; color: #92400e;">Data Description</h3>
                <div style="margin-bottom: 1.5rem;">
                    <h4 style="color: #92400e; margin-bottom: 0.5rem;">Food Flow Data Fields:</h4>
                    <table style="color: #78350f; font-size: 0.98rem; border-collapse: collapse;">
                        <tr><td style="padding: 2px 8px;"><code>origin</code></td><td>Origin county FIPS code</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>dest</code></td><td>Destination county FIPS code</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>origin_x</code>, <code>origin_y</code></td><td>Origin county coordinates</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>dest_x</code>, <code>dest_y</code></td><td>Destination county coordinates</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>predicted_value_original</code></td><td>kilotons of food shipped (GNN prediction)</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>exist_prob</code></td><td>Probability that this flow exists</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>sctg</code></td><td>Food category code (1-7)</td></tr>
                    </table>
                </div>
                <div style="margin-bottom: 1.5rem;">
                    <h4 style="color: #92400e; margin-bottom: 0.5rem;">Flow Information Fields:</h4>
                    <table style="color: #78350f; font-size: 0.98rem; border-collapse: collapse;">
                        <tr><td style="padding: 2px 8px;"><code>Origin_FIPS</code></td><td>Origin county FIPS code</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>Origin_County</code></td><td>Origin county name</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>Origin_State</code></td><td>Origin state name</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>Dest_FIPS</code></td><td>Destination county FIPS code</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>Dest_County</code></td><td>Destination county name</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>Dest_State</code></td><td>Destination state name</td></tr>
                        <tr><td style="padding: 2px 8px;"><code>kilotons_Shipped</code></td><td>Total kilotons shipped (summed over all trips between origin and destination)</td></tr>
                    </table>
                </div>
                <p style="color: #78350f; margin: 0; font-weight: 500;">
                    <strong>Data Source:</strong> FAF (Freight Analysis Framework) 2017 with Graph Neural Network predictions
                </p>
                <p style="color: #78350f; margin: 0.5rem 0 0 0; font-weight: 500;">
                    <strong>Note:</strong> Self-loops (origin = destination) have been filtered out to show only inter-county flows.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Filter Information Card ---
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #bae6fd;">
                <h3 style="margin: 0 0 1rem 0; color: #0c4a6e;">Filter Information</h3>
                <div style="color: #0369a1; line-height: 1.6;">
                    <p style="margin: 0 0 0.5rem 0;"><strong>Current Filters Applied:</strong></p>
                    <table style="margin: 0; padding-left: 0; color: #0369a1; font-size: 1rem;">
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Food Category:</strong></td>
                            <td>{cat}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Origin:</strong></td>
                            <td>{origin_lbl}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Destination:</strong></td>
                            <td>{dest_lbl}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Estimated Total Records:</strong></td>
                            <td>{len(filt):,}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px 12px 2px 0;"><strong>Estimated Total kilotons:</strong></td>
                            <td>{filt['predicted_value_original'].sum():,.1f}</td>
                        </tr>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

