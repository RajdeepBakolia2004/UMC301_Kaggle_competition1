import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping

# ---------------- Load data ----------------
train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")


# Target & ID columns
TARGET = "song_popularity"
IDCOL  = "id"

# Features: drop only ID and target
FEATURES = [c for c in train.columns if c not in [TARGET, IDCOL]]

# Fill missing values (simple median for speed)
train[FEATURES] = train[FEATURES].fillna(train[FEATURES].median())
test[FEATURES]  = test[FEATURES].fillna(train[FEATURES].median())

X = train[FEATURES]
y = train[TARGET]

# ---------------- Split for validation ----------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- LightGBM model ----------------
lgb = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=100, verbose=True)]
)

# ---------------- Validation AUC ----------------
val_preds = lgb.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_preds))

# ---------------- Train on full data & predict ----------------
lgb.fit(X, y)

test_preds = lgb.predict_proba(test[FEATURES])[:, 1]

# ---------------- Submission ----------------
submission = pd.DataFrame({
    IDCOL: test[IDCOL],
    TARGET: test_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")