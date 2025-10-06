import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping

# ---------------- Load data ----------------
train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")


TARGET = "song_popularity"
IDCOL  = "id"

# ---------------- Features ----------------
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
    X[c] = X[c].clip(0,1)
    X_test[c] = X_test[c].clip(0,1)

# ---------------- Feature Engineering ----------------
X["energy_valence"] = X["energy"] * X["audio_valence"]
X["dance_energy"] = X["danceability"] * X["energy"]
X["tempo_per_ms"] = X["tempo"] / (X["song_duration_ms"] + 1)
X["fast_tempo"] = (X["tempo"] > 120).astype(int)

X_test["energy_valence"] = X_test["energy"] * X_test["audio_valence"]
X_test["dance_energy"] = X_test["danceability"] * X_test["energy"]
X_test["tempo_per_ms"] = X_test["tempo"] / (X_test["song_duration_ms"] + 1)
X_test["fast_tempo"] = (X_test["tempo"] > 120).astype(int)

FEATURES = X.columns.tolist()  # update feature list

# ---------------- 5-Fold Stratified CV ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
val_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=128,
        subsample=0.9,
        colsample_bytree=0.9,
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

    val_pred = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold} AUC: {auc:.5f}")
    val_aucs.append(auc)

    test_preds += model.predict_proba(X_test)[:,1] / skf.n_splits

print(f"\nMean CV AUC: {np.mean(val_aucs):.5f}")

# ---------------- Submission ----------------
submission = pd.DataFrame({
    IDCOL: test[IDCOL],
    TARGET: test_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv done saved")
