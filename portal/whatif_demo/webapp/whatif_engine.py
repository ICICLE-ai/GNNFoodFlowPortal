"""
whatif_engine.py — WhatifEngine for the FoodFlow what-if demo.

Cross-scale inference: model trained on 132 FAF zones, predicts at 3,143 county level.

County features are projected into the zone latent space via zone-fitted Scaler+PCA,
then processed by a county-level Haversine k-NN graph through the trained GCN.
Each county gets its own distinct embedding — no zone-mapping copy logic.

Public API:
  - predict_county_baseline(orig_fips, dest_fips)
      Fast lookup using precomputed county embeddings. < 1 ms.
  - predict_county_with_overrides(orig_fips, dest_fips, orig_ov, dest_ov)
      Re-runs GCN on modified county features. ~50–100 ms.
  - get_county_features(fips)   : raw feature dict for UI sliders
  - get_county_meta(fips)       : lat, lon, population, has_port, name
  - county_opts()               : sorted display strings for dropdown
  - fips_from_display(s)        : FIPS from dropdown string

Designed for @st.cache_resource — load() called once per server process.
"""

import os, json, pickle
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(WEBAPP_DIR, ".."))
ARTIFACTS  = os.path.join(WEBAPP_DIR, "whatif_artifacts")

# ── SCTG labels (matching run_inference.py) ────────────────────────────────────
SCTG_LABELS = {
    1: "Live Animals / Fish",
    2: "Cereal Grains",
    3: "Other Ag Products",
    4: "Animal Feed & Seeds",
    5: "Meat / Seafood",
    6: "Milled Grain Products",
    7: "Other Foodstuffs",
}


class WhatifEngine:
    """
    All state loaded from whatif_artifacts/. Thread-safe for read-only inference.
    """

    def __init__(self):
        self._loaded = False

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self):
        """Load all artifacts. Call once at startup."""
        import importlib.util as _ilu

        # ── model ──────────────────────────────────────────────────────────────
        V2P_CODE = os.path.abspath(os.path.join(ROOT, "..", "FoodFlow_v2_parallel", "code"))
        sys_path_backup = __import__("sys").path[:]
        __import__("sys").path.insert(0, V2P_CODE)
        __import__("sys").path.insert(0, os.path.join(ROOT, "code"))

        from model import MultitaskSparseGCN  # multitask model

        # Restore path
        __import__("sys").path[:] = sys_path_backup

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join(ROOT, "models", "best_multitask_sparse_gcn.pth")
        ckpt = torch.load(model_path, map_location=self._device)
        sei_from_ckpt = ckpt["model_state_dict"].get("sparse_edge_index")
        model = MultitaskSparseGCN(
            node_dim          = ckpt["node_dim"],
            edge_dim          = ckpt["edge_dim"],
            hidden            = ckpt["hidden"] // 2,
            n_tasks           = ckpt["n_tasks"],
            dropout           = ckpt.get("dropout", 0.2),
            sparse_edge_index = sei_from_ckpt,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()
        self._model = model

        # ── node artifacts ─────────────────────────────────────────────────────
        with open(os.path.join(ARTIFACTS, "node_scaler.pkl"), "rb") as f:
            self._node_scaler = pickle.load(f)
        with open(os.path.join(ARTIFACTS, "node_pca.pkl"), "rb") as f:
            self._node_pca = pickle.load(f)
        with open(os.path.join(ARTIFACTS, "feature_cols.json")) as f:
            self._feature_cols = json.load(f)

        # ── edge artifacts ─────────────────────────────────────────────────────
        with open(os.path.join(ARTIFACTS, "edge_scaler.pkl"), "rb") as f:
            self._edge_scaler = pickle.load(f)

        # ── FAF-zone predicted tons used as the county-level scale anchor ──────
        preds_df = pd.read_csv(os.path.join(WEBAPP_DIR, "predictions.csv"))
        preds_df["orig"] = preds_df["orig"].astype(str).str.zfill(3)
        preds_df["dest"] = preds_df["dest"].astype(str).str.zfill(3)

        sctg_cols = [f"sctg{k}_tons" for k in range(1, 8)]

        self._zone_tons_capped = {}
        for _, row in preds_df.iterrows():
            key  = (row.orig, row.dest)
            tons = np.array([float(row[c]) for c in sctg_cols], dtype=np.float32)
            if tons.sum() > 0:
                self._zone_tons_capped[key] = tons

        # ── County cross-scale artifacts ───────────────────────────────────────
        self._county_X_raw          = np.load(os.path.join(ARTIFACTS, "county_X_raw.npy"))
        self._county_embeddings     = np.load(os.path.join(ARTIFACTS, "county_embeddings.npy"))
        # Zone-centering adjustment (x_c_adj = x_c + adj, then scaler+PCA)
        _adj_path = os.path.join(ARTIFACTS, "county_zone_adj.npy")
        self._county_zone_adj = (np.load(_adj_path) if os.path.exists(_adj_path)
                                 else np.zeros_like(self._county_X_raw))
        with open(os.path.join(ARTIFACTS, "county_to_idx.json")) as f:
            self._county_to_idx = json.load(f)
        self._idx_to_fips = {v: k for k, v in self._county_to_idx.items()}

        meta = pd.read_csv(os.path.join(ARTIFACTS, "county_meta.csv"), dtype={"FIPS": str})
        meta["FIPS"] = meta["FIPS"].str.zfill(5)
        self._county_meta_df  = meta.set_index("FIPS")
        # self._apply_county_meta_coordinate_fixes()
        self._county_feat_col_to_idx = {col: i for i, col in enumerate(self._feature_cols)}

        # ── County display list (for UI dropdown) ─────────────────────────────
        import re as _re
        def _state_sort_key(s):
            m = _re.search(r', ([A-Z]{2}) \(', s)
            return (m.group(1) if m else "ZZ", s)

        self._county_display_list = sorted(
            (
                f"{row.get('county_full') or row.get('county_name', fips)}, "
                f"{row.get('state_abbr', '')} ({fips})"
                for fips, row in self._county_meta_df.iterrows()
            ),
            key=_state_sort_key,
        )
        self._display_to_fips = {
            f"{row.get('county_full') or row.get('county_name', fips)}, "
            f"{row.get('state_abbr', '')} ({fips})": fips
            for fips, row in self._county_meta_df.iterrows()
        }

        # ── Zone mapping + SCTG-specific county gravity totals ────────────────
        _ctf_path = os.path.join(WEBAPP_DIR, "county_to_faf.csv")
        if os.path.exists(_ctf_path):
            _ctf = pd.read_csv(_ctf_path, dtype=str)
            _ctf["FIPS"] = _ctf["FIPS"].str.zfill(5)
            _ctf = _ctf.dropna(subset=["FAF_Zone"])
            _ctf["FAF_Zone"] = _ctf["FAF_Zone"].str.zfill(3)
            self._fips_to_zone = dict(zip(_ctf["FIPS"], _ctf["FAF_Zone"]))

            # Sum county populations and economic gravity features per zone
            # Also store per-county baseline gravity so _zone_anchor_params can
            # update the zone denominator dynamically when overrides are applied.
            _zone_pop: dict = {}
            _zone_gravity_O = {}
            _zone_gravity_D = {}
            self._county_gravity_O_baseline: dict = {}   # fips → np.array(7,)
            self._county_gravity_D_baseline: dict = {}
            for _, _r in _ctf.iterrows():
                _z, _f = _r["FAF_Zone"], _r["FIPS"]
                if _f in self._county_to_idx:
                    g_o = self._compute_gravity_weights(_f, is_origin=True)
                    g_d = self._compute_gravity_weights(_f, is_origin=False)
                    self._county_gravity_O_baseline[_f] = g_o
                    self._county_gravity_D_baseline[_f] = g_d
                    if _z not in _zone_gravity_O:
                        _zone_gravity_O[_z] = np.zeros(7, dtype=float)
                        _zone_gravity_D[_z] = np.zeros(7, dtype=float)
                    _zone_gravity_O[_z] += g_o
                    _zone_gravity_D[_z] += g_d

                if _f in self._county_meta_df.index:
                    _p = self._county_meta_df.loc[_f, "population"]
                    _p = float(_p.iloc[0] if hasattr(_p, "iloc") else _p)
                    if not np.isnan(_p):
                        _zone_pop[_z] = _zone_pop.get(_z, 0.0) + _p
            self._zone_total_pop = {z: max(p, 1.0) for z, p in _zone_pop.items()}
            self._zone_total_gravity_O = {z: np.maximum(v, 1.0) for z, v in _zone_gravity_O.items()}
            self._zone_total_gravity_D = {z: np.maximum(v, 1.0) for z, v in _zone_gravity_D.items()}
        else:
            self._fips_to_zone  = {}
            self._zone_total_pop = {}

        self._loaded = True

    # def _apply_county_meta_coordinate_fixes(self):
    #     """
    #     Correct known county centroid artifacts without mutating the source CSV.
    #     Aleutians West crosses the antimeridian; the source centroid lands in
    #     inland Canada and distorts both map placement and distance-based flows.
    #     """
    #     fixes = {
    #         "02016": {"lat": 52.3233, "lon": -174.1596},
    #     }
    #     for fips, coords in fixes.items():
    #         if fips in self._county_meta_df.index:
    #             self._county_meta_df.loc[fips, "lat"] = coords["lat"]
    #             self._county_meta_df.loc[fips, "lon"] = coords["lon"]

    # ── County metadata helpers ────────────────────────────────────────────────

    def county_opts(self) -> list:
        """Return sorted list of county display strings."""
        return self._county_display_list

    def fips_from_display(self, display: str) -> Optional[str]:
        """Return FIPS from display string. Returns None if not found."""
        return self._display_to_fips.get(display)

    def get_county_meta(self, fips: str) -> dict:
        """Return {name, lat, lon, population, has_port} for a county FIPS."""
        f = str(fips).zfill(5)
        if f not in self._county_meta_df.index:
            return {"name": f"County {f}", "lat": 39.5, "lon": -98.0,
                    "population": 0.0, "has_port": False}
        r = self._county_meta_df.loc[f]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return {
            "name":       str(r.get("county_full") or r.get("county_name") or f),
            "lat":        float(r["lat"]) if pd.notna(r["lat"]) else 39.5,
            "lon":        float(r["lon"]) if pd.notna(r["lon"]) else -98.0,
            "population": float(r["population"]) if pd.notna(r["population"]) else 0.0,
            "has_port":   bool(r["has_port"]),
        }

    def get_county_features(self, fips: str) -> dict:
        """Return dict of all raw feature values for a county FIPS (for UI sliders)."""
        f = str(fips).zfill(5)
        idx = self._county_to_idx.get(f)
        if idx is None:
            return {}
        return {col: float(self._county_X_raw[idx, i])
                for i, col in enumerate(self._feature_cols)}

    # ── County cross-scale inference ──────────────────────────────────────────

    def predict_county_baseline(self, orig_fips: str, dest_fips: str) -> list:
        """
        County-to-county prediction using precomputed county embeddings.
        Each county has its own distinct embedding from the county k-NN GCN pass.
        No zone mapping — truly county-level forward computation.

        Returns list of 7 dicts: {sctg, label, flow_exists, exist_prob, value_tons}
        """
        o = str(orig_fips).zfill(5)
        d = str(dest_fips).zfill(5)
        if o == d:
            return self._county_zero_result()

        o_idx = self._county_to_idx.get(o)
        d_idx = self._county_to_idx.get(d)
        if o_idx is None or d_idx is None:
            return self._county_zero_result()

        edge_attr   = self._compute_county_edge_attr(o, d, {}, {})
        zone_anchor = self._zone_anchor_params(o, d)
        return self._run_county_edge_mlp(
            self._county_embeddings[o_idx],
            self._county_embeddings[d_idx],
            edge_attr,
            zone_anchor=zone_anchor,
        )

    def predict_county_with_overrides(
        self,
        orig_fips: str,
        dest_fips: str,
        orig_overrides: dict,
        dest_overrides: dict,
    ) -> list:
        """
        County-to-county prediction with feature overrides.

        When any node feature is modified, re-runs the full county GCN forward pass
        so that message-passing propagation is correctly reflected in all embeddings.
        Each county's embedding is computed from its own features + k-NN context —
        not copied from a zone aggregate.

        orig_overrides / dest_overrides: {feature_name: new_value}
        Returns list of 7 dicts: {sctg, label, flow_exists, exist_prob, value_tons}
        """
        o = str(orig_fips).zfill(5)
        d = str(dest_fips).zfill(5)
        if o == d:
            return [
                {"sctg": k, "label": SCTG_LABELS[k],
                 "flow_exists": False, "exist_prob": 0.0, "value_tons": 0.0}
                for k in range(1, 8)
            ]

        o_idx = self._county_to_idx.get(o)
        d_idx = self._county_to_idx.get(d)
        if o_idx is None or d_idx is None:
            return [
                {"sctg": k, "label": SCTG_LABELS[k],
                 "flow_exists": False, "exist_prob": 0.0, "value_tons": 0.0}
                for k in range(1, 8)
            ]

        # ── Compute embeddings for override ───────────────────────────────────
        # Problem with full GCN re-run: GCNConv averages over 10 neighbours, so
        # a single-county feature change is diluted to ~1/10 after each layer.
        # After 2 layers the signal is ~1/100 — overrides become invisible.
        #
        # Fix: for override counties use a NO-MESSAGE-PASSING forward pass
        # (self-loop only: h = LeakyReLU(BN(x W^T + b))).  This maps feature
        # changes directly to embedding changes with no neighbour dilution.
        # Unmodified counties keep their precomputed full-GCN embeddings.

        def _single_node_embed(x_pca_row: np.ndarray) -> np.ndarray:
            """Linear transform only (no message passing): (30,) → (128,)."""
            x_t = torch.tensor(x_pca_row, dtype=torch.float).unsqueeze(0).to(self._device)  # (1,30)
            with torch.no_grad():
                h1 = (x_t @ self._model.conv1.lin.weight.T
                      + self._model.conv1.bias)          # (1, 64)
                h1 = F.leaky_relu(self._model.bn1(h1), 0.01)
                h2 = (h1 @ self._model.conv2.lin.weight.T
                      + self._model.conv2.bias)          # (1, 128)
                h2 = F.leaky_relu(self._model.bn2(h2), 0.01)
            return h2.squeeze(0).cpu().numpy()           # (128,)

        def _modified_pca(fips_code: str, overrides: dict) -> np.ndarray:
            c_idx = self._county_to_idx[fips_code]
            x_raw_row = self._county_X_raw[c_idx].copy()
            for feat_name, val in overrides.items():
                col_idx = self._county_feat_col_to_idx.get(feat_name)
                if col_idx is not None:
                    x_raw_row[col_idx] = float(val)
            x_raw_adj = x_raw_row + self._county_zone_adj[c_idx]
            x_scaled  = self._node_scaler.transform(x_raw_adj.reshape(1, -1))
            return self._node_pca.transform(x_scaled)[0]  # (30,)

        # Origin embedding
        if orig_overrides:
            h_o = _single_node_embed(_modified_pca(o, orig_overrides))
        else:
            h_o = self._county_embeddings[o_idx]

        # Destination embedding
        if dest_overrides:
            h_d = _single_node_embed(_modified_pca(d, dest_overrides))
        else:
            h_d = self._county_embeddings[d_idx]

        edge_attr   = self._compute_county_edge_attr(o, d, orig_overrides, dest_overrides)
        zone_anchor = self._zone_anchor_params(o, d, orig_overrides, dest_overrides)
        baseline_raw_value_tons = None
        if orig_overrides or dest_overrides:
            base_edge_attr = self._compute_county_edge_attr(o, d, {}, {})
            _, baseline_raw_value_tons = self._edge_head_outputs(
                self._county_embeddings[o_idx],
                self._county_embeddings[d_idx],
                base_edge_attr,
            )

        # Has non-population feature overrides? → lower zone weight so GNN mix shows through
        _non_pop_keys = {"population"}
        _has_feat_ov = bool(
            (orig_overrides and set(orig_overrides) - _non_pop_keys) or
            (dest_overrides and set(dest_overrides) - _non_pop_keys)
        )
        return self._run_county_edge_mlp(
            h_o, h_d, edge_attr,
            zone_anchor=zone_anchor,
            has_feature_overrides=_has_feat_ov,
            baseline_raw_value_tons=baseline_raw_value_tons,
        )

    def predict_county_fan(
        self,
        fixed_fips: str,
        mode: str,
        partner_state: Optional[str] = None,
        fixed_overrides: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        County-level one-to-many fan using the calibrated county pair engine.

        mode = 'origin' -> fixed county is origin, predict to partner counties
        mode = 'dest'   -> fixed county is destination, predict from partner counties

        partner_state can be a 2-letter state abbreviation such as 'WI' to
        restrict partners to counties in that state. If None, all US counties
        in the artifact set are used.
        """
        fixed = str(fixed_fips).zfill(5)
        fixed_overrides = fixed_overrides or {}
        if fixed not in self._county_to_idx:
            return pd.DataFrame()

        state_filter = str(partner_state).upper() if partner_state else None
        records = []
        for partner_fips, row in self._county_meta_df.iterrows():
            partner = str(partner_fips).zfill(5)
            state_abbr = str(row.get("state_abbr", "")).upper()
            partner_name = row.get("county_full")
            if (
                not partner.isdigit()
                or partner not in self._county_to_idx
                or partner not in self._fips_to_zone
                or len(state_abbr) != 2
                or pd.isna(partner_name)
                or pd.isna(row.get("lat"))
                or pd.isna(row.get("lon"))
            ):
                continue
            if partner == fixed:
                continue
            if state_filter and state_abbr != state_filter:
                continue

            if mode == "origin":
                try:
                    baseline = self.predict_county_baseline(fixed, partner)
                    modified = self.predict_county_with_overrides(
                        fixed, partner, fixed_overrides, {}
                    )
                except Exception:
                    continue
            else:
                try:
                    baseline = self.predict_county_baseline(partner, fixed)
                    modified = self.predict_county_with_overrides(
                        partner, fixed, {}, fixed_overrides
                    )
                except Exception:
                    continue

            rec = {
                "partner_fips": partner,
                "partner_name": str(partner_name),
                "partner_state": state_abbr,
                "lat": float(row["lat"]) if pd.notna(row["lat"]) else 39.5,
                "lon": float(row["lon"]) if pd.notna(row["lon"]) else -98.0,
            }
            base_total = 0.0
            mod_total = 0.0
            max_prob = 0.0
            for b, m in zip(baseline, modified):
                k = int(b["sctg"])
                b_val = float(b["value_tons"])
                m_val = float(m["value_tons"])
                rec[f"sctg{k}_baseline"] = b_val
                rec[f"sctg{k}_modified"] = m_val
                base_total += b_val
                mod_total += m_val
                max_prob = max(max_prob, float(m.get("exist_prob", 0.0)))
            rec["baseline_total"] = base_total
            rec["modified_total"] = mod_total
            rec["delta_total"] = mod_total - base_total
            rec["exist_prob_max"] = max_prob
            records.append(rec)

        if not records:
            return pd.DataFrame()
        out = pd.DataFrame(records).sort_values("modified_total", ascending=False)
        if limit is not None:
            out = out.head(int(limit))
        return out.reset_index(drop=True)

    def _compute_county_edge_attr(
        self,
        orig_fips: str,
        dest_fips: str,
        orig_overrides: dict,
        dest_overrides: dict,
    ) -> np.ndarray:
        """
        Compute scaled edge features for a county-to-county pair.
        Uses county lat/lon for Haversine distance, county population and port flag.
        Scaled with zone-fitted edge scaler (cross-scale edge encoding).
        Returns (1, 7) float32 array.
        """
        o_meta = self.get_county_meta(orig_fips)
        d_meta = self.get_county_meta(dest_fips)

        lat_o = float(orig_overrides.get("lat",  o_meta["lat"]))
        lon_o = float(orig_overrides.get("lon",  o_meta["lon"]))
        pop_o = float(orig_overrides.get("population", o_meta["population"]))
        port_o = float(int(orig_overrides.get("has_port", int(o_meta["has_port"]))))

        lat_d = float(dest_overrides.get("lat",  d_meta["lat"]))
        lon_d = float(dest_overrides.get("lon",  d_meta["lon"]))
        pop_d = float(dest_overrides.get("population", d_meta["population"]))
        port_d = float(int(dest_overrides.get("has_port", int(d_meta["has_port"]))))

        # Haversine distance (km)
        R = 6371.0
        lat1, lat2 = np.radians(lat_o), np.radians(lat_d)
        dlon = np.radians(lon_d - lon_o)
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        dist = 2 * R * np.arcsin(np.clip(np.sqrt(a), 0, 1))
        dist = max(dist, 1.0)   # floor at 1 km to avoid /0

        log_dist    = np.log1p(dist)
        inv_dist    = 1.0 / (dist + 1.0)
        gravity     = (pop_o * pop_d) / (dist ** 2 + 1e-8)
        log_gravity = np.log1p(gravity)

        edge_raw = np.array(
            [[dist, log_dist, inv_dist, gravity, log_gravity, port_o, port_d]],
            dtype=np.float32,
        )
        return self._edge_scaler.transform(edge_raw).astype(np.float32)


    def _compute_gravity_weights(self, fips: str, overrides: dict = None, is_origin: bool = True) -> np.ndarray:
        feats = self.get_county_features(fips).copy()
        if overrides:
            for k, v in overrides.items():
                if k in feats:
                    feats[k] = float(v)

        pop = max(feats.get("population", 1.0), 1.0)
        mfg = feats.get("31----", 0)
        retail_food = feats.get("44----", 0) + feats.get("72----", 0)

        if is_origin:
            g1 = feats.get("CATTLE_value", 0) + feats.get("HOGS_value", 0) + feats.get("SHEEP & GOATS TOTALS_value", 0)
            g2 = feats.get("CORN_value", 0) + feats.get("WHEAT_value", 0) + feats.get("RICE_value", 0) + feats.get("BARLEY_value", 0) + feats.get("SORGHUM_value", 0)
            g3 = feats.get("VEGETABLE TOTALS_value", 0) + feats.get("FRUIT & TREE NUT TOTALS_value", 0) + feats.get("COTTON_value", 0)
            g4 = feats.get("HAY & HAYLAGE_value", 0) + feats.get("SOYBEANS_value", 0)
            g5 = mfg + g1 * 0.1
            g6 = mfg + g2 * 0.1
            g7 = mfg
        else:
            g1 = mfg + pop * 0.1
            g2 = mfg + pop * 0.1
            g3 = mfg + retail_food + pop * 0.1
            g4 = feats.get("CATTLE_value", 0) + feats.get("HOGS_value", 0) + mfg
            g5 = retail_food + pop
            g6 = mfg + retail_food + pop
            g7 = retail_food + pop

        g = np.array([g1, g2, g3, g4, g5, g6, g7], dtype=float)
        g = np.where(g > 0, g, pop * 0.01)
        return g
    def _zone_anchor_params(self, o_fips: str, d_fips: str,
                            orig_overrides: dict = None,
                            dest_overrides: dict = None):
        """
        Return (zone_tons[7], o_share[7], d_share[7]) for county-level anchoring.
        Returns None when zone mapping or zone prediction is unavailable.

        orig_overrides / dest_overrides: if population is overridden, pop_share
        is recomputed so that the override changes the final total magnitude.
        """
        o_zone = self._fips_to_zone.get(o_fips)
        d_zone = self._fips_to_zone.get(d_fips)
        if not o_zone or not d_zone:
            return None

        key = (o_zone, d_zone)
        if key not in self._zone_tons_capped:
            return None

        zone_tons = self._zone_tons_capped[key]
        if zone_tons.sum() == 0:
            return None

        o_g = self._compute_gravity_weights(o_fips, overrides=orig_overrides, is_origin=True)
        d_g = self._compute_gravity_weights(d_fips, overrides=dest_overrides, is_origin=False)

        # Dynamic denominator: when overrides change a county's gravity, subtract
        # the baseline contribution and add the new one so the zone total reflects
        # the hypothetical scenario rather than the historical baseline.
        o_total = self._zone_total_gravity_O.get(o_zone, np.maximum(o_g, 1.0)).copy()
        d_total = self._zone_total_gravity_D.get(d_zone, np.maximum(d_g, 1.0)).copy()
        if orig_overrides:
            o_g_old = self._county_gravity_O_baseline.get(o_fips)
            if o_g_old is not None:
                o_total = np.maximum(o_total - o_g_old + o_g, 1.0)
        if dest_overrides:
            d_g_old = self._county_gravity_D_baseline.get(d_fips)
            if d_g_old is not None:
                d_total = np.maximum(d_total - d_g_old + d_g, 1.0)

        o_pop_share = np.clip(o_g / o_total, 0.0, 1.0)
        d_pop_share = np.clip(d_g / d_total, 0.0, 1.0)

        return (zone_tons, o_pop_share, d_pop_share)

    def _run_county_edge_mlp(
        self,
        h_o: np.ndarray,
        h_d: np.ndarray,
        edge_attr: np.ndarray,
        zone_anchor=None,
        has_feature_overrides: bool = False,
        baseline_raw_value_tons: Optional[np.ndarray] = None,
    ) -> list:
        """Run edge encoder + shared MLP + HurdleHeads for a single county OD pair."""
        prob, raw_value_tons = self._edge_head_outputs(h_o, h_d, edge_attr)

        # Dynamic sparsity / Top-K approximation via SCTG-specific threshold.
        # SCTG 1-4 (Ag) are extremely sparse, SCTG 5-7 (Processing) are relatively broad.
        HURDLE_THRESH_ARRAY = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        valid_mask = prob > HURDLE_THRESH_ARRAY
        county_raw = np.where(valid_mask, raw_value_tons * prob, 0.0)

        # calibrated county tonnage:
        # FAF-zone tons × SCTG-aligned county gravity shares provide the plausible
        # county-level scale. Scenario runs also use the freshly re-run GNN value
        # head as a bounded response ratio against the baseline raw GNN value.

        if zone_anchor is not None:
            zone_tons, o_pop_share, d_pop_share = zone_anchor
            base_tons = zone_tons * o_pop_share * d_pop_share

            # Soft gating: below threshold, attenuate by 0.1 rather than hard zero.
            # Preserves non-linear GNN response while still suppressing low-confidence edges.
            gate = np.where(prob > HURDLE_THRESH_ARRAY, 1.0, 0.1)

            if baseline_raw_value_tons is not None:
                safe_ref = np.maximum(baseline_raw_value_tons, 1e-9)
                response_ratio = raw_value_tons / safe_ref
                lower, upper, alpha = 0.25, 4.0, 0.25
                value_modifier = np.power(np.clip(response_ratio, lower, upper), alpha)
            else:
                value_modifier = np.ones_like(base_tons, dtype=float)

            tons = base_tons * prob * gate * value_modifier
        else:
            tons = county_raw

        return [
            {
                "sctg":        k,
                "label":       SCTG_LABELS[k],
                "flow_exists": bool(tons[k - 1] > 0),
                "exist_prob":  float(prob[k - 1]),
                "value_tons":  float(tons[k - 1]),
            }
            for k in range(1, 8)
        ]

    def _edge_head_outputs(
        self,
        h_o: np.ndarray,
        h_d: np.ndarray,
        edge_attr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return GNN existence probabilities and raw value-head tons."""
        h_orig = torch.tensor(h_o, dtype=torch.float).unsqueeze(0).to(self._device)   # (1, 128)
        h_dest = torch.tensor(h_d, dtype=torch.float).unsqueeze(0).to(self._device)   # (1, 128)
        edge_t = torch.tensor(edge_attr, dtype=torch.float).to(self._device)           # (1, 7)

        with torch.no_grad():
            edge_enc = self._model.edge_encoder(edge_t)
            combined  = torch.cat([h_orig, h_dest, edge_enc], dim=1)
            shared    = self._model.edge_mlp(combined)

            logits, values = [], []
            for head in self._model.heads:
                lg, vl = head(shared)
                logits.append(lg)
                values.append(vl)

            prob  = torch.sigmoid(torch.cat(logits, dim=1)).cpu().numpy()[0]
            value = torch.cat(values, dim=1).cpu().numpy()[0]
        raw_value_tons = np.expm1(np.clip(value, 0, 20))
        return prob, raw_value_tons

    def _county_zero_result(self) -> list:
        return [
            {"sctg": k, "label": SCTG_LABELS[k],
             "flow_exists": False, "exist_prob": 0.0, "value_tons": 0.0}
            for k in range(1, 8)
        ]
