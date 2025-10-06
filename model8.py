import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping

# ---------------- Load data ----------------
train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")

TARGET = "song_popularity"
IDCOL  = "id"
FEATURES = [c for c in train.columns if c not in [TARGET, IDCOL]]

# ---------------- Impute missing values ----------------
train[FEATURES] = train[FEATURES].fillna(train[FEATURES].median())
test[FEATURES]  = test[FEATURES].fillna(train[FEATURES].median())

X = train[FEATURES].copy()
y = train[TARGET].copy()
X_test = test[FEATURES].copy()

# ---------------- Clip probability-like features ----------------
prob_cols = ["acousticness","danceability","energy",
             "instrumentalness","liveness","audio_valence"]
for c in prob_cols:
    X[c] = X[c].clip(0, 1)
    X_test[c] = X_test[c].clip(0, 1)

# ---------------- Feature Engineering ----------------
def add_features(df):
    df["energy_valence"] = df["energy"] * df["audio_valence"]
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["tempo_per_ms"] = df["tempo"] / (df["song_duration_ms"] + 1)
    df["fast_tempo"] = (df["tempo"] > 120).astype(int)
    
    # Ratios
    df["energy_dance_ratio"] = df["energy"] / (df["danceability"] + 1e-6)
    df["valence_energy_ratio"] = df["audio_valence"] / (df["energy"] + 1e-6)
    
    # Polynomials
    df["energy_sq"] = df["energy"] ** 2
    df["dance_valence"] = df["danceability"] * df["audio_valence"]
    
    # Bins
    df["tempo_bin"] = pd.qcut(df["tempo"], 5, labels=False, duplicates="drop")
    df["loudness_bin"] = pd.qcut(df["loudness"], 5, labels=False, duplicates="drop")
    df["duration_bin"] = pd.qcut(df["song_duration_ms"], 5, labels=False, duplicates="drop")

add_features(X)
add_features(X_test)

# ---------------- Target Encoding ----------------
X_temp = X.copy()
X_temp[TARGET] = y

for col in ["key", "audio_mode", "time_signature"]:
    mean_target = X_temp.groupby(col)[TARGET].mean()
    X[col + "_te"] = X[col].map(mean_target)
    X_test[col + "_te"] = X_test[col].map(mean_target)

FEATURES = X.columns.tolist()

# ---------------- 5-Fold Stratified CV ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
val_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=256,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold} AUC: {auc:.5f}")
    val_aucs.append(auc)

    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

print(f"\nMean CV AUC: {np.mean(val_aucs):.5f}")

# ---------------- Submission ----------------
submission = pd.DataFrame({
    IDCOL: test[IDCOL],
    TARGET: test_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")