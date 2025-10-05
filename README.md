# 🎵 UMC301_Kaggle_Competition1  
**Predicting Song Popularity using Machine Learning (AIML Mini-Project)**  

---

## 📘 Overview
This repository contains my submission for the **UMC 301 AIML Kaggle Competition** — a regression-based challenge to **predict the popularity of songs** using various musical and acoustic features such as *danceability, valence, energy, tempo, loudness,* and *acousticness*.  

The project explores multiple models, ensemble strategies, and data preprocessing techniques to achieve high performance on the Kaggle leaderboard.

---

## 🧠 Objective
To predict the **popularity score** of songs using supervised machine learning models trained on numerical and categorical song attributes.

---

## 📊 Dataset Description
Each song in the dataset includes the following attributes:

| Feature | Description |
|----------|--------------|
| **Danceability** | Suitability of a track for dancing based on rhythm, tempo, and beat strength. |
| **Valence** | Musical positiveness or emotional tone (happy/sad). |
| **Energy** | Perceptual measure of intensity and activity. |
| **Tempo** | Speed of the song in beats per minute (BPM). |
| **Loudness** | Average loudness in decibels (dB). |
| **Speechiness** | Presence of spoken words. |
| **Instrumentalness** | Likelihood the track is instrumental. |
| **Liveness** | Probability the song was recorded live. |
| **Acousticness** | Confidence measure of whether a track is acoustic. |
| **Key** | Estimated key (0 = C, 1 = C#, etc.). |
| **Mode** | Modality of the track (1 = Major, 0 = Minor). |
| **Duration** | Duration of the song in milliseconds. |
| **Time Signature** | Beats per bar in the track. |

---

## ⚙️ Methods and Models
The following models were developed and evaluated during the competition:

1. **Random Forest Regressor**  
2. **XGBoost Regressor**  
3. **LightGBM Regressor**  
4. **CatBoost Regressor**  
5. **Stacked/Ensemble Models** (various combinations)

A total of **8 submissions** were made on Kaggle, and **2 final models** were selected for final leaderboard submission based on validation and public scores.

---

## 🧩 Workflow
1. **Data Preprocessing & Imputation**
   - Handled missing values using mean, median, and KNN imputation.
   - Standardized and normalized numerical columns.
   - Encoded categorical columns (Key, Mode, Time Signature).

2. **Feature Engineering**
   - Feature scaling, correlation filtering, and interaction features tested.

3. **Model Training & Validation**
   - Split data into train/validation sets.
   - Performed hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
   - Evaluated models on validation RMSE and Kaggle public leaderboard.

4. **Ensembling**
   - Combined best-performing models using weighted average and stacking.

---

## 🧾 Repository Structure
