# BelceeriinDoroitolV2 – Land Degradation Evaluation in Mongolia

This repository contains my **Bachelor’s thesis project** on evaluating pastureland degradation in Hustai National Park, Mongolia, using **multispectral drone imagery** and **image processing techniques**.

The system classifies orthomosaic images into **grass**, **weeds**, and **bare soil**, enabling automated assessment of land condition. The workflow reduces reliance on manual field surveys, potentially saving **hundreds of man-hours** in ecosystem research.

---

## 📂 Repository Overview

* **`experiments/`** – Jupyter notebooks and scripts for testing segmentation and classification methods.
* **`reflectance_msavi_threshold/`** – Core implementation of reflectance calibration, vegetation index (MSAVI) computation, and threshold-based segmentation.
* **`gabor_src/`** – Early texture-based methods (Gabor filters) by previous students.
* **`outputs/`** – Processed results and classification visualizations.
* **`data/`** – References to raw and processed datasets (not included due to size).
* **`tutorials/`** – Supplementary material (e.g., Landsat segmentation, MONAI medical segmentation tutorial).

---

## 🚀 Key Features

* Automated **image alignment** and preprocessing.
* Reflectance normalization using DJI Phantom 4 multispectral drone sun sensor data.
* **MSAVI-based vegetation index analysis** for degraded land detection.
* Experimentation with:

  * KMeans clustering
  * Random Forests
  * Object-Based Image Analysis (OBIA)
  * Early deep learning models (FastAI, ViTs)
* Compatible with **QGIS** and **OpenDroneMap (ODM)** outputs.

---

## 📊 Results

* Clear separation of **healthy vegetation vs. degraded land**.
* Thresholding + MSAVI normalization improved robustness across acquisition dates.
* Prototype demonstrated feasibility of replacing large-scale manual surveys.

Example workflow:

```bash
python cli_segmentation.py \
    --input data/raw/Drone_Orthomosaics/Best_2024.tif \
    --output outputs/results/Best_2024_segmented.tif \
    --method msavi-threshold
```

---

## 📚 References

* [Batnyambuu Dashpurev, Lukas Lehnert et al. – *A cost-effective method for monitoring land degradation*](https://www.sciencedirect.com/science/article/pii/S1470160X21009961)
* [E. Celikkan et al. – *WeedsGalore Dataset*](https://arxiv.org/abs/2502.13103)

---

## ✅ TODO

* [ ] Merge internship progress (`Dadlaga_2024`).
* [ ] Refactor repo into a modular structure.
* [ ] Publish reproducible demo notebook with sample dataset.
