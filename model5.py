# ===================== 1. Imports =====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ===================== 2. Load data =====================
train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")


X = train.drop(["id", "song_popularity"], axis=1)
y = train["song_popularity"]
test_ids = test["id"]
X_test = test.drop(["id"], axis=1)

# ===================== 3. Preprocess =====================
num_cols = X.columns

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols)
])

# ===================== 4. Hyper-parameter search space =====================
param_dist = {
    "xgb__max_depth":        [3, 4, 5, 6, 8],
    "xgb__learning_rate":    [0.01, 0.02, 0.05, 0.1],
    "xgb__n_estimators":     [800, 1200, 2000, 3000],
    "xgb__subsample":        [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.6, 0.8, 1.0],
    "xgb__min_child_weight": [1, 3, 5],
    "xgb__gamma":            [0, 1, 3],
    "xgb__reg_lambda":       [1, 5, 10],
    "xgb__reg_alpha":        [0, 1, 5]
}

pipe = Pipeline([
    ("prep", preprocess),
    ("xgb", XGBClassifier(
        tree_method="hist",
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="auc",       # built-in AUC metric
        n_jobs=-1,
        random_state=42
    ))
])

# ===================== 5. Randomized Search =====================
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=40,                # increase for more thorough search
    scoring="roc_auc",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X, y)

print("\nBest params:", search.best_params_)
print("Best CV AUC:", search.best_score_)

# ===================== 6. Fit best model on full data =====================
best_model = search.best_estimator_
best_model.fit(X, y)   # train on entire training set

# ===================== 7. Predict & Save Submission =====================
probs = best_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"id": test_ids, "song_popularity": probs})
submission.to_csv("submission.csv", index=False)
print("submission.csv written.")