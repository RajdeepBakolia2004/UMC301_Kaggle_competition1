import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# ------------------ Load ------------------
train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")

# ------------------ Target & IDs ------------------
y = train["song_popularity"]
X = train.drop(["id", "song_popularity"], axis=1)
test_ids = test["id"]
X_test = test.drop(["id"], axis=1)

# ------------------ Data Cleaning ------------------
# Clip bounded audio features to [0,1]
bounded_cols = ["acousticness","danceability","energy",
                "instrumentalness","liveness","audio_valence"]

for col in bounded_cols:
    X[col] = X[col].clip(0,1)
    X_test[col] = X_test[col].clip(0,1)

# Convert duration from ms → minutes
X["song_duration_min"] = X["song_duration_ms"] / 60000
X_test["song_duration_min"] = X_test["song_duration_ms"] / 60000
X.drop("song_duration_ms", axis=1, inplace=True)
X_test.drop("song_duration_ms", axis=1, inplace=True)

# ------------------ Feature Engineering ------------------
# Loudness bins
X["loudness_bin"] = pd.qcut(X["loudness"], q=5, labels=False, duplicates="drop")
X_test["loudness_bin"] = pd.qcut(X_test["loudness"], q=5, labels=False, duplicates="drop")

# Tempo bins (slow, medium, fast, very fast)
X["tempo_bin"] = pd.cut(X["tempo"], bins=[0,90,120,200,300], labels=False)
X_test["tempo_bin"] = pd.cut(X_test["tempo"], bins=[0,90,120,200,300], labels=False)

# Interaction features
X["energy_dance"] = X["energy"] * X["danceability"]
X_test["energy_dance"] = X_test["energy"] * X_test["danceability"]

X["valence_energy"] = X["audio_valence"] * X["energy"]
X_test["valence_energy"] = X_test["audio_valence"] * X_test["energy"]

# ------------------ Handle Missing ------------------
# Simple median fill for any remaining NaNs
X = X.fillna(X.median())
X_test = X_test.fillna(X.median())

# ------------------ Categorical Features ------------------
categorical_features = ["key","audio_mode","time_signature",
                        "loudness_bin","tempo_bin"]

# Ensure categorical cols are integers
for col in categorical_features:
    X[col] = X[col].astype("int").astype("str")  # safer as strings
    X_test[col] = X_test[col].astype("int").astype("str")

# ------------------ Train/Validation Split ------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------ Class Weights ------------------
pos_weight = (len(y) - sum(y)) / sum(y)  # imbalance adjustment

# ------------------ CatBoost Model ------------------
cat = CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=5,
    eval_metric="AUC",
    loss_function="Logloss",
    cat_features=categorical_features,
    class_weights=[1, pos_weight],
    random_seed=42,
    verbose=200,
    thread_count=-1
)

# ------------------ Train & Validate ------------------
cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)

val_pred = cat.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_pred))

# ------------------ Final Training on Full Data ------------------
cat.fit(X, y)

# ------------------ Submission ------------------
probs = cat.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"id": test_ids, "song_popularity": probs})
submission.to_csv("submission.csv", index=False)
print("submission.csv written.")