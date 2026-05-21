# GNN Food Flow Portal

The GNN Food Flow Portal is a Streamlit web app for exploring county-level U.S. food-flow predictions from the [GNN FoodFlow Model](https://github.com/GeoDS/GNNFoodFlow). Version 1.2.0 includes the original parallel SCTG-specific model exploration workflow plus a new multi-task GNN what-if portal for one-to-one and one-to-many county scenario analysis.

**Live portal:** https://gnnfoodflowportal.pods.icicleai.tapis.io/

**Tags:** Software, AI4CI, Visual-Analytics, Digital-Agriculture, Smart-Foodsheds

For guidance on the Tutorials, How-To Guides, Explanation, and References organization used in this README, see [Diátaxis](https://diataxis.fr/).

### License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## References

- [GNN FoodFlow Model](https://github.com/GeoDS/GNNFoodFlow): related model repository.
- [GNN Food Flow Portal source code](https://github.com/ICICLE-ai/GNNFoodFlowPortal): GitHub repository for this portal.
- [Live GNN Food Flow Portal](https://gnnfoodflowportal.pods.icicleai.tapis.io/): deployed Streamlit portal.
- [FAF5.6.1 SCTG commodity flow data](https://faf.ornl.gov/faf5/): Freight Analysis Framework data used for food-flow modeling.
- [SCTG commodity code reference](https://bhs.econ.census.gov/bhsphpext/brdsearch/scs_code.html): descriptions of Standard Classification of Transported Goods categories.
- [U.S. Census cartographic boundary files](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html): county and state shapefiles used for map visualization.
- [ICICLE Training Catalog](https://icicle-ai.github.io/training-catalog/): ICICLE component documentation hub.

## Acknowledgements

*National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)*

## Issue reporting

Please report issues, bugs, documentation gaps, or feature requests through the [GNN Food Flow Portal GitHub Issues page](https://github.com/ICICLE-ai/GNNFoodFlowPortal/issues).

---

# Tutorials

## Explore Parallel Model Food Flows

Use this tutorial to inspect precomputed county-to-county food-flow predictions.

1. Open the [live portal](https://gnnfoodflowportal.pods.icicleai.tapis.io/) or run the portal locally.
2. Start the app and move from the welcome story into the main portal.
3. Select **Parallel Model Map**.
4. In the sidebar, choose a food category from SCTG 1-7.
5. Select an origin county, a destination county, or choose **All** for broader patterns.
6. Adjust the number of top links and the flow-width scaling mode.
7. Hover over map arcs to inspect origin, destination, FIPS codes, and predicted kilotons shipped.
8. Open **Parallel Model Download** to export filtered rows or origin-destination summary statistics as CSV.

## Run a Multi-Task What-If Scenario

Use this tutorial to compare baseline food-flow estimates with scenario-adjusted estimates from the vendored multi-task GNN demo assets.

1. Open the main portal.
2. Select **Multi-task One-to-One** to compare one origin county with one destination county, or select **Multi-task One-to-Many** to compare one focus county with many partners.
3. Choose the county or counties for the scenario.
4. Adjust regional feature values in the scenario form.
5. Click **Run One-to-One** or **Run One-to-Many**.
6. Review the baseline, modified, and delta results.
7. Download the scenario CSV for downstream analysis.

---

# How-To Guides

## Run the Portal Locally

Follow these steps to set up and run the GNN Food Flow Portal from a local checkout.

### 1. Clone the repository

```bash
git clone https://github.com/ICICLE-ai/GNNFoodFlowPortal.git
cd GNNFoodFlowPortal
```

### 2. Create a Python environment

Python 3.11 is recommended for the current 1.2.0 configuration.

```bash
conda create -n gnnfoodflow python=3.11
conda activate gnnfoodflow
```

### 3. Install dependencies

```bash
cd portal
pip install -r requirements.txt
```

The what-if workflow depends on PyTorch, PyTorch Geometric, and scikit-learn. If your platform requires a custom PyTorch wheel, install the correct PyTorch package for your operating system and accelerator before installing the remaining requirements.

### 4. Run Streamlit

```bash
streamlit run app.py
```

### 5. Open the portal

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in a browser and use the portal navigation to switch between the parallel model and multi-task what-if workflows.

## Use the Parallel Model Download Page

1. Select **Parallel Model Download**.
2. Choose the SCTG category and any origin or destination filters in the sidebar.
3. Download **Current Filtered Data** for the rows currently used by the map.
4. Download **Summary Statistics** for origin-destination totals with county and state labels.

## Use the What-If Export

1. Run either a one-to-one or one-to-many scenario.
2. Review the baseline, modified, and delta outputs.
3. Use the download button on the results page to export the scenario comparison as CSV.

## Build the Docker Image

The portal includes a Dockerfile for container deployment.

```bash
cd portal
docker build -t gnn-food-flow-portal .
docker run --rm -p 8501:8501 gnn-food-flow-portal
```

For Tapis pod deployments, the image uses CPU-compatible PyTorch packages because the current pods do not provide GPU support. The build also uses a slim Python 3.11 base image, excludes local-only files through `portal/.dockerignore`, and cleans caches and generated bytecode to reduce image size.

---

# Explanation

## What Version 1.2.0 Adds

Version 1.2.0 incorporates a new FoodFlow portal workflow alongside the existing static parallel-model map and download tools.

- **Parallel Model Map:** explores precomputed SCTG-specific county-to-county food-flow predictions.
- **Parallel Model Guide:** explains how to use the static model filters, map, hover details, and export workflow.
- **Parallel Model Download:** exports filtered model rows and origin-destination summary statistics.
- **Multi-task One-to-One:** runs a scenario for a selected origin-destination county pair and compares baseline and modified flows.
- **Multi-task One-to-Many:** runs a scenario for one focus county against many partner counties and summarizes partner-level changes.
- **Vendored what-if assets:** includes multi-task model artifacts, feature metadata, county mappings, predictions, a model checkpoint, and inference code under `portal/whatif_demo`.

## Deployment Notes

The May 2026 Tapis deployment was rebuilt to include the new multi-task what-if workflows. Deployment updates include:

- The what-if workflow adds `torch`, `torch-geometric`, and `scikit-learn` to the portal dependency set.
- The container uses CPU-only PyTorch packages for the current non-GPU pod environment.
- The Docker build uses `python:3.11-slim-bookworm`, build-context exclusions, and cleanup steps to reduce image size.
- Security-related packages were updated, including GitPython, urllib3, pillow, tornado, and Streamlit.
- Container runtime fixes resolved path handling issues in `app.py` and `whatif_tabs.py`.
- The older parallel map workflow may load more slowly than the new scenario tabs in some environments, but it remains functional.

## Data and Artifacts

The repository includes the data and artifacts needed by the portal.

- `portal/cleaned_data`: precomputed parallel SCTG-specific model outputs.
- `portal/whatif_demo`: multi-task what-if artifacts, model checkpoint, metadata, and inference code.
- `portal/data`: county metadata and shapefile resources.
- `portal/files`: supporting files, including the project PDF and FAF metadata.
- `portal/image`: portal images and branding assets.

## Data Fields

The parallel model workflow uses county-level origin-destination rows with fields such as:

- `origin` and `dest`: county FIPS codes.
- `origin_x`, `origin_y`, `dest_x`, and `dest_y`: county coordinates.
- `predicted_value_original`: estimated kilotons of food shipped.
- `exist_prob`: probability that the predicted flow exists.
- `sctg`: food category code.

## Downstream Use

- Spatial forecasting of trade changes under policy shifts.
- Identifying critical counties for supply-chain resilience.
- Exploring food-flow sensitivity under county-level feature changes.
- Generating scenario comparison tables for research and planning.

## Out-of-Scope Use

- Real-time food trade forecasting.
- Non-U.S. geographic settings without retraining.
- Operational routing, procurement, or emergency decisions without external validation.

## Bias, Risks, and Limitations

- **Bias:** Model predictions depend on historical FAF data and may not reflect unexpected future disruptions, such as disasters, pandemics, or sudden policy changes.
- **Limitations:** Static parallel-model prediction is limited to predefined food SCTG categories 1-7.
- **Scenario uncertainty:** What-if outputs are model-based estimates and should be interpreted as exploratory analysis, not guaranteed causal effects.
- **Data quality:** Results assume the accuracy and coverage of FAF flow data, county metadata, economic indicators, and model artifacts.
