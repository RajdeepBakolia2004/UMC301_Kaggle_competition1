from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd


# ---------------- Load data ----------------
train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")

TARGET = "song_popularity"
IDCOL  = "id"

# ---------------- Features & Imputation ----------------
FEATURES = [c for c in train.columns if c not in [TARGET, IDCOL]]
train[FEATURES] = train[FEATURES].fillna(train[FEATURES].median())
test[FEATURES]  = test[FEATURES].fillna(train[FEATURES].median())

X = train[FEATURES].copy()
y = train[TARGET].copy()
X_test = test[FEATURES].copy()

# ---------------- Clip 0–1 range columns ----------------
prob_cols = ["acousticness","danceability","energy",
             "instrumentalness","liveness","audio_valence"]
for c in prob_cols:
    X[c] = X[c].clip(0,1)
    X_test[c] = X_test[c].clip(0,1)

# ---------------- Feature Engineering ----------------
X["energy_valence"] = X["energy"] * X["audio_valence"]
X["dance_energy"]   = X["danceability"] * X["energy"]
X["tempo_per_ms"]   = X["tempo"] / (X["song_duration_ms"] + 1)
X["fast_tempo"]     = (X["tempo"] > 120).astype(int)

X_test["energy_valence"] = X_test["energy"] * X_test["audio_valence"]
X_test["dance_energy"]   = X_test["danceability"] * X_test["energy"]
X_test["tempo_per_ms"]   = X_test["tempo"] / (X_test["song_duration_ms"] + 1)
X_test["fast_tempo"]     = (X_test["tempo"] > 120).astype(int)

FEATURES = X.columns.tolist()

# X, y : your preprocessed features/labels

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

for train_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        tree_method="hist",
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="auc",        
    )

    # Fit **without** eval_metric or early_stopping in .fit()
    model.fit(X_tr, y_tr, verbose=False)

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Fold AUC: {auc:.4f}")
    oof_preds[val_idx] = val_pred

print("OOF AUC:", roc_auc_score(y, oof_preds))

# Train final model on full data
final_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    tree_method="hist",
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="auc"
)
final_model.fit(X, y, verbose=False)

# Predict probabilities for the correctly processed test set
test_preds = final_model.predict_proba(X_test)[:, 1]

# Create the submission file with the correct ID and column name
submission = pd.DataFrame({
    IDCOL: test[IDCOL],
    TARGET: test_preds
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")