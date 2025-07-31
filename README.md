# Emotions in Motion 🌟

**Mood Prediction Engine** using Machine Learning & Deep Learning  
Built for: PG Certificate in Data Science & AI — IIT Roorkee × CloudxLab

---

## 🚀 Overview

Emotions in Motion is an AI-powered system that predicts human mood — *Good, Neutral, or Bad* — using everyday behavior data (activity, stress, energy, and time of day). The vision is clear: emotionally aware technology that understands and gently supports us.

---

## 🎯 What I Built

- **FitLife Dataset** (100K+ records) capturing lifestyle and emotional signals  
- **Data Processing Pipeline**:
  - Cleaning & leakage prevention  
  - Mood conversion (1–10 scale → 3 categories)  
  - One-Hot Encoding → 317 features  
  - SMOTE oversampling for fairness  
  - StandardScaler for numeric normalization  

---

## 🧠 Models Explored

- **Random Forest** — baseline (72.4% accuracy), interpretable  
- **XGBoost** — improved prediction of negative moods (74.1%) + SHAP interpretability  
- **Deep Neural Network** — best generalization (74.2%), with dropout, batch-norm & early stopping  

**Model Comparison:** DNN > XGBoost > Random Forest — each presents unique strengths.

---

## 🔍 Interpretability & Insights

- 🎯 **Random Forest Feature Importance** shows key predictors like *Mood Before* and *Duration*  
- 🔎 **SHAP (XGBoost)** visualizes how each feature influences predictions — it adds credibility and clarity

---

## 💡 Key Learnings

- Human mood can be **quantified and predicted** using behavioral patterns  
- **SMOTE** ensures balanced training across mood classes  
- **SHAP** builds transparency and trust in model decisions  
- **Deep Learning** offers best overall performance for emotional pattern detection

---

## 🔧 Installation & Usage

```bash
git clone https://github.com/Prince5104/Capstone_Project…
cd Emotions-in-Motion
pip install -r requirements.txt
python train_and_evaluate
