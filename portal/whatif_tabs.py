"""Embeddable multi-task what-if tabs for the GNN Food Flow Portal.

This module intentionally does not call st.set_page_config(). It loads the
vendored Plan D demo engine/artifacts from portal/whatif_demo so the portal can
be packaged and deployed without depending on sibling workspace folders.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


PORTAL_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PORTAL_DIR.parents[2]
VENDORED_WEBAPP_DIR = PORTAL_DIR / "whatif_demo" / "webapp"
LEGACY_DEMO_WEBAPP_DIR = WORKSPACE_DIR / "FoodFlow_multitask_planD_demo" / "webapp"


SCTG_LABELS = {
    1: "Live Animals / Fish",
    2: "Cereal Grains",
    3: "Other Ag Products",
    4: "Animal Feed & Seeds",
    5: "Meat / Seafood",
    6: "Milled Grain Products",
    7: "Other Foodstuffs",
}

NAICS_LABELS = {
    "31----": "Manufacturing",
    "42----": "Wholesale Trade",
    "44----": "Retail Trade",
    "72----": "Accommodation & Food Services",
    "48----": "Transportation & Warehousing",
    "11----": "Agriculture & Forestry",
    "23----": "Construction",
    "62----": "Health Care",
    "52----": "Finance & Insurance",
    "54----": "Professional Services",
}

TOP_CROPS = [
    "CATTLE_value",
    "HAY_value",
    "HAYLAGE_value",
    "CORN_value",
    "GOATS_value",
    "VEGETABLE TOTALS_value",
    "SOYBEANS_value",
    "HOGS_value",
    "SHEEP_value",
]

CROP_LABELS = {col: col.replace("_value", "").title() for col in TOP_CROPS}

CROSS_SCALE_NOTICE = (
    "<strong>Note:</strong> "
    "County-level data are reference estimates derived from a larger regional model. "
    "We recommend using these totals for trend analysis, ranking, and comparing different "
    "scenarios, rather than as precise absolute statistics, as individual county values may be inflated. "
    "<br><strong>How it works:</strong> Our model uses multi-task AI model (GNN) to capture broad trends, "
    "then distributes regional totals based on county-specific weights. This ensures the outputs "
    "respond effectively to changes while staying within a realistic scale."
)


@st.cache_resource(show_spinner="Loading FoodFlow what-if model...")
def get_whatif_engine():
    demo_webapp_dir = _demo_webapp_dir()
    if not demo_webapp_dir.exists():
        raise FileNotFoundError(
            f"Missing vendored what-if workspace: {VENDORED_WEBAPP_DIR}. "
            "Expected portal/whatif_demo/webapp with whatif_engine.py, "
            "whatif_artifacts/, predictions.csv, and county_to_faf.csv."
        )
    demo_path = str(demo_webapp_dir)
    if demo_path not in sys.path:
        sys.path.insert(0, demo_path)
    from whatif_engine import WhatifEngine

    engine = WhatifEngine()
    engine.load()
    return engine


def _demo_webapp_dir() -> Path:
    return VENDORED_WEBAPP_DIR if VENDORED_WEBAPP_DIR.exists() else LEGACY_DEMO_WEBAPP_DIR


@st.cache_resource(show_spinner=False)
def load_whatif_metadata() -> dict:
    """Load lightweight county metadata/features without importing torch/model code."""
    demo_webapp_dir = _demo_webapp_dir()
    artifacts = demo_webapp_dir / "whatif_artifacts"
    if not artifacts.exists():
        raise FileNotFoundError(f"Missing what-if artifacts: {artifacts}")

    meta = pd.read_csv(artifacts / "county_meta.csv", dtype={"FIPS": str})
    meta["FIPS"] = meta["FIPS"].str.zfill(5)
    meta = meta.set_index("FIPS")

    with open(artifacts / "county_to_idx.json") as f:
        county_to_idx = json.load(f)
    with open(artifacts / "feature_cols.json") as f:
        feature_cols = json.load(f)

    county_x_raw = np.load(artifacts / "county_X_raw.npy")
    feat_col_to_idx = {col: i for i, col in enumerate(feature_cols)}

    def state_sort_key(display: str) -> tuple:
        state = display.rsplit(", ", 1)[-1].split(" (", 1)[0] if ", " in display else "ZZ"
        return (state, display)

    display_list = sorted(
        (
            f"{row.get('county_full') or row.get('county_name', fips)}, "
            f"{row.get('state_abbr', '')} ({fips})"
            for fips, row in meta.iterrows()
            if fips in county_to_idx
        ),
        key=state_sort_key,
    )
    display_to_fips = {
        f"{row.get('county_full') or row.get('county_name', fips)}, "
        f"{row.get('state_abbr', '')} ({fips})": fips
        for fips, row in meta.iterrows()
        if fips in county_to_idx
    }

    return {
        "county_opts": display_list,
        "display_to_fips": display_to_fips,
        "meta": meta,
        "county_to_idx": county_to_idx,
        "county_x_raw": county_x_raw,
        "feature_cols": feature_cols,
        "feat_col_to_idx": feat_col_to_idx,
    }


def _engine_or_stop():
    try:
        return get_whatif_engine()
    except Exception as exc:
        st.error(f"What-if model could not be loaded: {exc}")
        st.stop()


def _metadata_or_stop() -> dict:
    try:
        return load_whatif_metadata()
    except Exception as exc:
        st.error(f"What-if metadata could not be loaded: {exc}")
        st.stop()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .whatif-kicker {
            font-size: .72rem;
            font-weight: 800;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .25rem;
        }
        .whatif-notice {
            background: #fefce8;
            border: 1px solid #fde68a;
            border-radius: 10px;
            color: #78350f;
            font-size: .86rem;
            padding: .75rem 1rem;
            margin: .75rem 0 1rem;
        }
        .whatif-flow-card {
            align-items: center;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            margin-bottom: .45rem;
            padding: .65rem .85rem;
        }
        .whatif-flow-card.baseline { border-top: 3px solid #93c5fd; }
        .whatif-flow-card.modified { border-top: 3px solid #86efac; }
        .whatif-flow-name {
            color: #1f2937;
            font-size: .86rem;
            font-weight: 700;
        }
        .whatif-flow-value {
            color: #166534;
            font-size: .88rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .whatif-delta {
            border-radius: 6px;
            display: inline-block;
            font-size: .7rem;
            font-weight: 800;
            margin-left: .45rem;
            padding: .12rem .35rem;
        }
        .whatif-delta.up { background: #dcfce7; color: #166534; }
        .whatif-delta.down { background: #fee2e2; color: #991b1b; }
        .multitask-banner {
            background: linear-gradient(135deg, #14213d 0%, #006b3c 55%, #c8a900 100%);
            border-radius: 14px;
            color: white;
            margin-bottom: 1rem;
            padding: 1.1rem 1.25rem;
        }
        .multitask-banner h2 {
            color: white !important;
            font-size: 1.55rem;
            line-height: 1.2;
            margin: 0;
        }
        .multitask-banner p {
            color: rgba(255,255,255,.86) !important;
            font-size: .9rem;
            margin: .35rem 0 0;
        }
        .multitask-pill {
            background: rgba(255,255,255,.16);
            border: 1px solid rgba(255,255,255,.28);
            border-radius: 999px;
            color: white;
            display: inline-flex;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .08em;
            margin-bottom: .5rem;
            padding: .25rem .65rem;
            text-transform: uppercase;
        }
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #006b3c 0%, #0f8a52 55%, #c8a900 100%) !important;
            background-color: #006b3c !important;
            border: 1px solid rgba(255,255,255,.25) !important;
            border-radius: 10px !important;
            box-shadow: 0 6px 18px rgba(0, 107, 60, .22) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            min-height: 2.75rem;
            --primary-color: #006b3c !important;
            --secondary-background-color: #006b3c !important;
        }
        div[data-testid="stFormSubmitButton"] > button *,
        div[data-testid="stFormSubmitButton"] button * {
            color: #ffffff !important;
        }
        div[data-testid="stFormSubmitButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(135deg, #005730 0%, #0b7645 55%, #b99b00 100%) !important;
            background-color: #005730 !important;
            border-color: rgba(255,255,255,.35) !important;
            box-shadow: 0 8px 22px rgba(0, 107, 60, .28) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stFormSubmitButton"] > button:focus,
        div[data-testid="stFormSubmitButton"] button:focus,
        div[data-testid="stFormSubmitButton"] > button:active,
        div[data-testid="stFormSubmitButton"] button:active {
            background: linear-gradient(135deg, #005730 0%, #0b7645 55%, #b99b00 100%) !important;
            background-color: #005730 !important;
            border-color: #c8a900 !important;
            color: #ffffff !important;
            outline: 2px solid rgba(200,169,0,.25) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_multitask_banner(mode: str) -> None:
    st.markdown(
        f"""
        <div class="multitask-banner">
            <div class="multitask-pill">Multi-task Model</div>
            <h2>{mode}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _state_choices(county_opts: list[str]) -> list[str]:
    states = {
        opt.rsplit(", ", 1)[-1].split(" (", 1)[0]
        for opt in county_opts
        if ", " in opt and len(opt.rsplit(", ", 1)[-1].split(" (", 1)[0]) == 2
    }
    return sorted(states)


def _state_county_opts(county_opts: list[str], state: str) -> list[str]:
    return [
        opt
        for opt in county_opts
        if opt.rsplit(", ", 1)[-1].split(" (", 1)[0] == state
    ]


def _default_idx(options: list[str], fips: str) -> int:
    return next((i for i, opt in enumerate(options) if fips in opt), 0)


def _fips_from_display(meta_cache: dict, display: str) -> Optional[str]:
    return meta_cache["display_to_fips"].get(display)


def _county_meta(meta_cache: dict, fips: str) -> dict:
    f = str(fips).zfill(5)
    if f not in meta_cache["meta"].index:
        return {
            "name": f,
            "lat": 39.5,
            "lon": -98.0,
            "population": 0.0,
            "has_port": False,
        }
    row = meta_cache["meta"].loc[f]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {
        "name": row.get("county_full") or row.get("county_name", f),
        "lat": float(row.get("lat", 39.5)),
        "lon": float(row.get("lon", -98.0)),
        "population": float(row.get("population", 0.0)),
        "has_port": bool(row.get("has_port", 0.0)),
    }


def _county_features(meta_cache: dict, fips: str) -> dict:
    f = str(fips).zfill(5)
    idx = meta_cache["county_to_idx"].get(f)
    if idx is None:
        return {}
    row = meta_cache["county_x_raw"][int(idx)]
    return {
        col: float(row[i])
        for col, i in meta_cache["feat_col_to_idx"].items()
    }


def _fmt_ktons(ktons: float) -> str:
    tons = float(ktons) * 1000.0
    if tons >= 1_000_000:
        return f"{tons / 1_000_000:.2f}M t"
    if tons >= 1_000:
        return f"{tons:,.0f} t"
    return f"{tons:.0f} t"


def _pct_row(label: str, base_val: float, key: str) -> Optional[float]:
    base_display = float(round(base_val))
    value_key = f"whatif_value_{key}"
    mode_key = f"whatif_mode_{key}"
    modes = ["Manual value", "-50%", "-25%", "Baseline", "+25%", "+50%"]
    pct_by_mode = {
        "-50%": -0.50,
        "-25%": -0.25,
        "Baseline": 0.0,
        "+25%": 0.25,
        "+50%": 0.50,
    }

    value_col, mode_col = st.columns([1.5, 1])
    with value_col:
        manual_val = st.number_input(
            label,
            min_value=0.0,
            value=base_display,
            step=max(1.0, round(base_display * 0.01, 0)),
            format="%.0f",
            key=value_key,
        )
    with mode_col:
        mode = st.selectbox(
            "Change",
            modes,
            index=0,
            key=mode_key,
            help="Sometimes single value might not lead to significant change in results. Change multiple relevant values to see more changes. And destination side change will be more responsive to the food transportation due to changes in its demand.",
        )

    if mode in pct_by_mode:
        new_val = max(0.0, base_display * (1 + pct_by_mode[mode]))
    else:
        new_val = float(manual_val)

    if abs(new_val - base_display) > 0.5:
        st.caption(f"Default: {int(base_display):,}")
        return new_val
    return None


def _override_panel(meta_cache: dict, fips: str, side: str) -> dict:
    overrides = {}
    feats = _county_features(meta_cache, fips)
    meta = _county_meta(meta_cache, fips)
    key_prefix = f"{side}_{str(fips).zfill(5)}"

    with st.expander("Population & Port", expanded=False):
        cur_pop = float(feats.get("population", meta["population"]))
        cur_port = bool(feats.get("has_port", int(meta["has_port"])))
        new_pop = _pct_row("Population", cur_pop, f"{key_prefix}_population")
        if new_pop is not None:
            overrides["population"] = new_pop

        new_port = st.checkbox("Has major port", value=cur_port, key=f"whatif_port_{key_prefix}")
        if int(new_port) != int(cur_port):
            st.caption(f"Default: {'Yes' if cur_port else 'No'}")
            overrides["has_port"] = float(int(new_port))

    with st.expander("Employment by Sector", expanded=False):
        st.caption("Employees by NAICS sector.")
        for col, label in NAICS_LABELS.items():
            new_val = _pct_row(label, float(feats.get(col, 0.0)), f"{key_prefix}_{col}")
            if new_val is not None:
                overrides[col] = new_val

    with st.expander("Agricultural Production", expanded=False):
        st.caption("Selected crop and livestock production values.")
        for col in TOP_CROPS:
            new_val = _pct_row(CROP_LABELS[col], float(feats.get(col, 0.0)), f"{key_prefix}_{col}")
            if new_val is not None:
                overrides[col] = new_val

    return overrides


def _result_signature(*parts) -> tuple:
    normalized = []
    for part in parts:
        if isinstance(part, dict):
            normalized.append(tuple(sorted((key, float(value)) for key, value in part.items())))
        else:
            normalized.append(str(part))
    return tuple(normalized)


def _render_notice() -> None:
    st.markdown(f'<div class="whatif-notice">{CROSS_SCALE_NOTICE}</div>', unsafe_allow_html=True)


def _render_disclaimer_footer() -> None:
    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#64748b;font-size:.78rem;line-height:1.7;border-top:1px solid #e2e8f0;padding-top:1.25rem">
        <strong>Disclaimer</strong> &nbsp;·&nbsp;
        Predictions are generated by a Graph Neural Network trained on 2017 freight flow data and are
        intended for exploratory and research purposes only. They do not constitute forecasts of actual
        commodity movements. Feature overrides simulate hypothetical regional conditions and do not
        account for supply chain constraints, policy changes, or economic dynamics outside the training
        data. Treat all outputs as model estimates with inherent uncertainty.
        <br><br>
        <strong>References</strong><br>
        [1] U.S. Department of Agriculture, National Agricultural Statistics Service. 2017. Quick Stats Agricultural Database.
        <a href="https://quickstats.nass.usda.gov/" target="_blank">https://quickstats.nass.usda.gov/</a><br>
        [2] US Census Bureau. 2021. SAIPE State and County Estimates for 2017.
        <a href="https://www.census.gov/data/datasets/2017/demo/saipe/2017-state-and-county.html" target="_blank">https://www.census.gov/data/datasets/2017/demo/saipe/2017-state-and-county.html</a><br>
        [3] US Census Bureau. 2023. County Population Totals: 2010-2019.
        <a href="https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html" target="_blank">https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html</a><br>
        [4] U.S. Census Bureau. 2019. 2017 County Business Patterns.
        <a href="https://www.census.gov/data/datasets/2017/econ/cbp/2017-cbp.html" target="_blank">https://www.census.gov/data/datasets/2017/econ/cbp/2017-cbp.html</a><br>
        [5] U.S. Department of Transportation, Bureau of Transportation Statistics. 2022. Principal Ports [GIS Dataset].
        <a href="https://geodata.bts.gov/datasets/usdot::principalports/" target="_blank">https://geodata.bts.gov/datasets/usdot::principalports/</a><br>
        [6] U.S. Department of Transportation, Bureau of Transportation Statistics and U.S. Department of Commerce, U.S. Census Bureau. 2020. 2017 Commodity Flow Survey Final Tables.
        <a href="https://trid.trb.org/View/2586794" target="_blank">https://trid.trb.org/View/2586794</a>;
        <a href="https://www2.census.gov/programs-surveys/cfs/data/2017/" target="_blank">https://www2.census.gov/programs-surveys/cfs/data/2017/</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pair_results(baseline: list[dict], modified: list[dict], orig_name: str, dest_name: str) -> None:
    base_total = sum(float(row["value_tons"]) for row in baseline)
    mod_total = sum(float(row["value_tons"]) for row in modified)
    delta = mod_total - base_total
    delta_pct = (delta / base_total * 100.0) if base_total > 0 else np.nan

    st.markdown(f"### {orig_name} to {dest_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline total", _fmt_ktons(base_total))
    c2.metric("Modified total", _fmt_ktons(mod_total))
    c3.metric("Change", _fmt_ktons(delta), f"{delta_pct:+.1f}%" if delta_pct == delta_pct else None)
    c4.metric(
        "Categories affected",
        f"{sum(abs(m['value_tons'] - b['value_tons']) * 1000 >= 0.5 for b, m in zip(baseline, modified))} / 7",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Baseline")
        for row in baseline:
            st.markdown(
                f"""
                <div class="whatif-flow-card baseline">
                    <span class="whatif-flow-name">SCTG {row['sctg']:02d} {row['label']}</span>
                    <span class="whatif-flow-value">{_fmt_ktons(row['value_tons'])}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("#### Modified")
        for row, base in zip(modified, baseline):
            diff = float(row["value_tons"]) - float(base["value_tons"])
            badge = ""
            if abs(diff) * 1000 >= 0.5:
                cls = "up" if diff >= 0 else "down"
                pct = diff / float(base["value_tons"]) * 100.0 if base["value_tons"] > 0 else np.nan
                pct_text = f"{pct:+.1f}%" if pct == pct else "new"
                badge = f'<span class="whatif-delta {cls}">{pct_text}</span>'
            st.markdown(
                f"""
                <div class="whatif-flow-card modified">
                    <span class="whatif-flow-name">SCTG {row['sctg']:02d} {row['label']}{badge}</span>
                    <span class="whatif-flow-value">{_fmt_ktons(row['value_tons'])}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Show comparison table", expanded=False):
        table = []
        for base, mod in zip(baseline, modified):
            diff = float(mod["value_tons"]) - float(base["value_tons"])
            table.append(
                {
                    "Category": f"SCTG {base['sctg']:02d} - {base['label']}",
                    "Baseline": _fmt_ktons(base["value_tons"]),
                    "Modified": _fmt_ktons(mod["value_tons"]),
                    "Delta": _fmt_ktons(diff),
                }
            )
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)


def _render_pair_map(meta_cache: dict, orig_fips: str, dest_fips: str, orig_name: str, dest_name: str) -> None:
    orig = _county_meta(meta_cache, orig_fips)
    dest = _county_meta(meta_cache, dest_fips)
    if not all(pd.notna(v) for v in [orig["lat"], orig["lon"], dest["lat"], dest["lon"]]):
        return

    arc_df = pd.DataFrame(
        [{"slat": orig["lat"], "slon": orig["lon"], "dlat": dest["lat"], "dlon": dest["lon"]}]
    )
    pts_df = pd.DataFrame(
        [
            {"lat": orig["lat"], "lon": orig["lon"], "label": orig_name, "color": [249, 115, 22, 220]},
            {"lat": dest["lat"], "lon": dest["lon"], "label": dest_name, "color": [59, 130, 246, 220]},
        ]
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[
                pdk.Layer(
                    "ArcLayer",
                    arc_df,
                    get_source_position=["slon", "slat"],
                    get_target_position=["dlon", "dlat"],
                    get_width=5,
                    get_tilt=15,
                    get_source_color=[249, 115, 22, 220],
                    get_target_color=[59, 130, 246, 220],
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    pts_df,
                    get_position=["lon", "lat"],
                    get_fill_color="color",
                    get_radius=35000,
                    pickable=True,
                ),
            ],
            initial_view_state=pdk.ViewState(
                latitude=(float(orig["lat"]) + float(dest["lat"])) / 2,
                longitude=(float(orig["lon"]) + float(dest["lon"])) / 2,
                zoom=4,
                pitch=15,
            ),
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"text": "{label}"},
        ),
        height=360,
        use_container_width=True,
    )


def render_one_to_one_tab() -> None:
    _inject_styles()
    meta_cache = _metadata_or_stop()

    _render_multitask_banner("One-to-One What-if Portal")
    st.caption("Pick an origin and destination county, modify regional features, then compare baseline and scenario flows.")
    _render_notice()

    county_opts = meta_cache["county_opts"]
    states = _state_choices(county_opts)

    col_orig, col_dest = st.columns(2)
    with col_orig:
        orig_state = st.selectbox(
            "Origin state",
            states,
            index=states.index("CA") if "CA" in states else 0,
            key="whatif_pair_origin_state",
        )
        orig_opts = _state_county_opts(county_opts, orig_state)
        orig_sel = st.selectbox(
            "Origin county",
            orig_opts,
            index=_default_idx(orig_opts, "06037"),
            key="whatif_pair_origin_county",
        )
    with col_dest:
        dest_state = st.selectbox(
            "Destination state",
            states,
            index=states.index("OH") if "OH" in states else 0,
            key="whatif_pair_dest_state",
        )
        dest_opts = _state_county_opts(county_opts, dest_state)
        dest_sel = st.selectbox(
            "Destination county",
            dest_opts,
            index=_default_idx(dest_opts, "39049"),
            key="whatif_pair_dest_county",
        )

    orig_fips = _fips_from_display(meta_cache, orig_sel)
    dest_fips = _fips_from_display(meta_cache, dest_sel)
    orig_name = orig_sel.rsplit(" (", 1)[0]
    dest_name = dest_sel.rsplit(" (", 1)[0]

    with st.form("whatif_pair_form"):
        st.markdown("### Adjust Features")
        st.caption("Feature changes are staged in this form. The app runs only when you click Run One-to-One.")
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{orig_name}**")
            orig_overrides = _override_panel(meta_cache, orig_fips, "pair_origin")
        with right:
            st.markdown(f"**{dest_name}**")
            dest_overrides = _override_panel(meta_cache, dest_fips, "pair_dest")
        run_pair = st.form_submit_button("Run One-to-One", type="primary", use_container_width=True)

    signature = _result_signature(orig_fips, dest_fips, orig_overrides, dest_overrides)
    if run_pair:
        engine = _engine_or_stop()
        with st.spinner("Running one-to-one scenario..."):
            baseline = engine.predict_county_baseline(orig_fips, dest_fips)
            modified = engine.predict_county_with_overrides(orig_fips, dest_fips, orig_overrides, dest_overrides)
        st.session_state["whatif_pair_results"] = (baseline, modified, orig_name, dest_name, orig_fips, dest_fips)
        st.session_state["whatif_pair_signature"] = signature

    if st.session_state.get("whatif_pair_signature") == signature and "whatif_pair_results" in st.session_state:
        baseline, modified, orig_name, dest_name, orig_fips, dest_fips = st.session_state["whatif_pair_results"]
        _render_pair_results(baseline, modified, orig_name, dest_name)
        _render_pair_map(meta_cache, orig_fips, dest_fips, orig_name, dest_name)

        export_df = pd.DataFrame(
            [
                {
                    "origin": orig_name,
                    "destination": dest_name,
                    "sctg": base["sctg"],
                    "category": base["label"],
                    "baseline_short_tons": base["value_tons"] * 1000,
                    "modified_short_tons": mod["value_tons"] * 1000,
                    "delta_short_tons": (mod["value_tons"] - base["value_tons"]) * 1000,
                }
                for base, mod in zip(baseline, modified)
            ]
        )
        st.download_button(
            "Download One-to-One CSV",
            data=export_df.to_csv(index=False),
            file_name="whatif_one_to_one.csv",
            mime="text/csv",
        )
    else:
        st.info("Adjust features if needed, then run the one-to-one scenario.")
    _render_disclaimer_footer()


def render_one_to_many_tab() -> None:
    _inject_styles()
    meta_cache = _metadata_or_stop()

    _render_multitask_banner("One-to-Many What-if Portal")
    st.caption("Focus one county and estimate flows to or from many partner counties.")
    _render_notice()

    county_opts = meta_cache["county_opts"]
    states = _state_choices(county_opts)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        fixed_state = st.selectbox(
            "Focus state",
            states,
            index=states.index("CA") if "CA" in states else 0,
            key="whatif_fan_state",
        )
        fixed_opts = _state_county_opts(county_opts, fixed_state)
        fixed_sel = st.selectbox(
            "Focus county",
            fixed_opts,
            index=_default_idx(fixed_opts, "06037"),
            key="whatif_fan_county",
        )
    with c2:
        direction = st.radio(
            "Direction",
            ["Origin to many destinations", "Many origins to destination"],
            key="whatif_fan_direction",
        )
        mode = "origin" if direction.startswith("Origin") else "dest"
    with c3:
        scope = st.radio(
            "Partner scope",
            ["Whole country", "One state"],
            index=1,
            key="whatif_fan_scope",
        )
        partner_state = None
        if scope == "One state":
            partner_state = st.selectbox(
                "Partner state",
                states,
                index=states.index("CA") if "CA" in states else 0,
                key="whatif_fan_partner_state",
            )

    top_n = st.slider("Show top partners", min_value=10, max_value=100, value=25, step=5, key="whatif_fan_top_n")
    fixed_fips = _fips_from_display(meta_cache, fixed_sel)
    fixed_name = fixed_sel.rsplit(" (", 1)[0]

    with st.form("whatif_fan_form"):
        st.markdown(f"### Scenario changes for {fixed_name}")
        st.caption("Feature changes are staged in this form. The app runs only when you click Run One-to-Many.")
        fixed_overrides = _override_panel(meta_cache, fixed_fips, "fan_fixed")
        run_fan = st.form_submit_button("Run One-to-Many", type="primary", use_container_width=True)

    signature = _result_signature(fixed_fips, mode, partner_state or "ALL", fixed_overrides, top_n)

    if run_fan:
        engine = _engine_or_stop()
        with st.spinner("Running one-to-many comparison..."):
            fan_df = engine.predict_county_fan(
                fixed_fips=fixed_fips,
                mode=mode,
                partner_state=partner_state,
                fixed_overrides=fixed_overrides,
                limit=top_n,
            )
        st.session_state["whatif_fan_results"] = fan_df
        st.session_state["whatif_fan_signature"] = signature
        st.session_state["whatif_fan_labels"] = (fixed_name, mode, partner_state or "Whole country")

    if st.session_state.get("whatif_fan_signature") == signature and "whatif_fan_results" in st.session_state:
        fan_df = st.session_state["whatif_fan_results"]
        if fan_df.empty:
            st.warning("No partner counties found for this selection.")
            _render_disclaimer_footer()
            return

        fixed_name, mode, scope_label = st.session_state["whatif_fan_labels"]
        base_total = float(fan_df["baseline_total"].sum())
        mod_total = float(fan_df["modified_total"].sum())
        delta = mod_total - base_total

        st.markdown(f"### {fixed_name} | {scope_label}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline total", _fmt_ktons(base_total))
        m2.metric("Modified total", _fmt_ktons(mod_total))
        m3.metric("Change", _fmt_ktons(delta), f"{(delta / base_total * 100):+.1f}%" if base_total > 0 else None)
        m4.metric("Active partners", f"{int((fan_df['modified_total'] > 0).sum())} / {len(fan_df)}")

        fixed_meta = _county_meta(meta_cache, fixed_fips)
        map_df = fan_df.copy()
        max_total = max(float(map_df["modified_total"].max()), 1.0)
        map_df["radius"] = 8000 + 42000 * np.sqrt(map_df["modified_total"].clip(lower=0) / max_total)
        map_df["color"] = map_df["delta_total"].apply(
            lambda value: [22, 163, 74, 175] if value >= 0 else [185, 28, 28, 175]
        )
        arcs = map_df.head(10).assign(
            slat=fixed_meta["lat"],
            slon=fixed_meta["lon"],
            dlat=lambda frame: frame["lat"],
            dlon=lambda frame: frame["lon"],
        )
        fixed_df = pd.DataFrame(
            [
                {
                    "lat": fixed_meta["lat"],
                    "lon": fixed_meta["lon"],
                    "label": fixed_name,
                    "color": [249, 115, 22, 230],
                    "radius": 50000,
                }
            ]
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[
                    pdk.Layer(
                        "ArcLayer",
                        arcs,
                        get_source_position=["slon", "slat"],
                        get_target_position=["dlon", "dlat"],
                        get_width=3,
                        get_source_color=[249, 115, 22, 200],
                        get_target_color=[22, 163, 74, 190],
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        map_df,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius="radius",
                        pickable=True,
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        fixed_df,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius="radius",
                        pickable=True,
                    ),
                ],
                initial_view_state=pdk.ViewState(
                    latitude=float(fixed_meta["lat"]),
                    longitude=float(fixed_meta["lon"]),
                    zoom=3,
                    pitch=15,
                ),
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                tooltip={"text": "{partner_name} {partner_state}"},
            ),
            height=390,
            use_container_width=True,
        )

        table_df = fan_df[["partner_name", "partner_state", "baseline_total", "modified_total", "delta_total"]].copy()
        table_df["Baseline"] = table_df["baseline_total"].map(_fmt_ktons)
        table_df["Modified"] = table_df["modified_total"].map(_fmt_ktons)
        table_df["Delta"] = table_df["delta_total"].map(_fmt_ktons)
        table_df = table_df.rename(columns={"partner_name": "County", "partner_state": "State"})
        st.dataframe(table_df[["County", "State", "Baseline", "Modified", "Delta"]], hide_index=True, use_container_width=True)

        export_df = fan_df[
            ["partner_fips", "partner_name", "partner_state", "baseline_total", "modified_total", "delta_total"]
        ].copy()
        export_df = export_df.rename(
            columns={
                "partner_name": "county",
                "partner_state": "state",
                "baseline_total": "baseline_short_tons",
                "modified_total": "modified_short_tons",
                "delta_total": "delta_short_tons",
            }
        )
        for col in ["baseline_short_tons", "modified_short_tons", "delta_short_tons"]:
            export_df[col] = export_df[col] * 1000
        st.download_button(
            "Download One-to-Many CSV",
            data=export_df.to_csv(index=False),
            file_name="whatif_one_to_many.csv",
            mime="text/csv",
        )
    else:
        st.info("Choose a focus county and partner scope, then run the one-to-many scenario.")
    _render_disclaimer_footer()
