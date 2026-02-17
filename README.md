## 🖼️ Visualizations

### Pose Estimation & Angle Tracking
The system extracts real-time gait angles (Hip, Knee, Ankle) by calculating the vectors between detected keypoints. Below is an example of the pose estimation overlay on a patient video:

![Gait Analysis Pose Estimation](Screenshot%202026-02-18%20002732.png)

*Note: The green text overlays indicate calculated joint angles used for clinical assessment.*

---


# Video-based Gait Analysis for Disease Prediction System

This repository contains a Python-based pipeline for disease prediction and gait assessment using video analysis. It leverages pose estimation and machine learning to identify key gait parameters and classify neuromuscular and joint-related conditions based on patient walking videos.

---

## 🧠 Overview

This project processes 100+ patient videos (from [Mendeley Gait Dataset](https://data.mendeley.com/datasets/44pfnysy89/1)) and intends to:

- Extract gait parameters from videos using **MediaPipe Pose**
- Clean and normalize time-series joint trajectory data
- Train classifiers to detect probable **gait-affecting diseases**
- Interpret model outputs with **SHAP analysis**
- Track and visualize **rehabilitation progress** relative to healthy references

---

## 🛠️ Features

- 🔍 **Pose Estimation**: Uses MediaPipe Pose to extract joint positions (hip, knee, ankle, etc.)
- 📊 **Gait Parameter Extraction**: Computes step/stride length, cadence, joint range of motion, double/single support time, swing/stance ratio (some features are to be added)
- 🧼 **Data Cleaning**: Handles missing landmarks with interpolation, smooths noise using Savitzky–Golay filtering
- 🧠 **Model Training**: Trains ML models (Random Forest, XGBoost, SVM, Logistic Regression) for disease classification
- 🔎 **Explainability**: Uses **SHAP (SHapley Additive Explanations)** to identify key biomarkers for each disease class
- 📈 **Progress Tracker**: Compares individual gait metrics against healthy reference baselines across sessions(not complete)

---

## 📁 Dataset

- Dataset used: [Mendeley Gait Dataset v1](https://data.mendeley.com/datasets/44pfnysy89/1)
- Contains walking videos of patients with:
  - Knee Osteoarthritis
  - Parkinson’s Disease
  - Nemaline Myopathy
  - Control (Healthy)

---

## 🔧 Requirements

```bash
pip install -r requirements.txt
