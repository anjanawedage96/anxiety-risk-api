# Prediction Model

A machine‑learning pipeline to predict anxiety risk levels (Low, Medium, High) from survey data, achieving a test macro‑F1 of 0.852 and overall accuracy of 0.800.

## Overview
- **Dataset:** Demographic, lifestyle, physiological, and self‑reported measures for 11,000+ individuals  
- **Target:** Anxiety level binned into Low (≤3), Medium (4–7), High (>7) :contentReference[oaicite:9]{index=9}  
- **Goal:** Build and compare classification models to maximize macro‑F1

## Key Steps
1. **Setup & Data Load**  
   Load CSV, set constants, inspect features (age, sleep, activity, caffeine/alcohol intake, symptoms, etc.) :contentReference[oaicite:10]{index=10}  
2. **Feature Engineering**  
   – Interaction terms (e.g. sleep×stress, alcohol per activity)  
   – Composite scores (unhealthy_score, symptom_severity)  
   – Bucketing (age groups, sleep categories, activity levels) :contentReference[oaicite:11]{index=11}  
3. **Target Binning**  
   Convert 1–10 anxiety scale into three ordered classes to address imbalance :contentReference[oaicite:12]{index=12}  
4. **Train/Test Split**  
   Stratified 80/20 split to preserve class proportions :contentReference[oaicite:13]{index=13}  
5. **Global Benchmark**  
   Compare Logistic Regression, Random Forest, XGBoost (with/without class weighting and SMOTE variants) via 5‑fold CV on macro‑F1 :contentReference[oaicite:14]{index=14}  
   – **Best:** RandomForestClassifier(n_estimators=200, class_weight='balanced') (CV macro‑F1 ≈ 0.846)

## Evaluation
- **Test Results (RF, balanced):**  
  – Macro‑F1: 0.852  
  – Accuracy: 0.800  
  – “High” class essentially perfect (F1 = 0.995; only 2 misclassifications) :contentReference[oaicite:15]{index=15}  

## Hierarchical Model (Optional)
- **Stage 1:** High vs. Rest  
- **Stage 2:** Low vs. Medium on remaining samples  
- **Outcome:** Macro‑F1(all) = 0.854 (+0.002), Macro‑F1(L+M) = 0.784 (marginal gain) :contentReference[oaicite:16]{index=16}  

## Conclusion & Next Steps
- The flat RF (balanced) pipeline is simplest and strongest.  
- Unless prioritizing one specific class, stick with the single‑stage model.  
- **Potential improvements:**  
  1. Fine‑tune class weights or probability thresholds  
  2. Prune/rework features with negative importance  
  3. Explore ordinal/regression‑then‑round baselines  
  4. Package pipeline with preprocessing and reproducible metric logging :contentReference[oaicite:17]{index=17}  
