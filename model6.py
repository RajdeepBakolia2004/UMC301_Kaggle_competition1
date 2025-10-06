import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# ------------------ Load ------------------
train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")

X = train.drop(["id", "song_popularity"], axis=1)
y = train["song_popularity"]
test_ids = test["id"]
X_test = test.drop(["id"], axis=1)

# ------------------ Quick preprocessing ------------------
num_cols = X.columns
prep = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols)
])

# ------------------ Fast CatBoost ------------------
# Fewer trees + moderate depth keeps training <1 min on CPU
cat = CatBoostClassifier(
    iterations=500,        # 500 trees
    depth=6,               # moderate tree depth
    learning_rate=0.05,    # slightly higher for faster convergence
    l2_leaf_reg=3,
    eval_metric="AUC",
    loss_function="Logloss",
    verbose=False,
    thread_count=-1,
    random_seed=42
)

pipe = Pipeline([
    ("prep", prep),
    ("cat", cat)
])

# ------------------ Train & Validate quickly ------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipe.fit(X_train, y_train)

val_pred = pipe.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_pred))

# ------------------ Final submission ------------------
pipe.fit(X, y)
probs = pipe.predict_proba(X_test)[:, 1]
pd.DataFrame({"id": test_ids, "song_popularity": probs}) \
  .to_csv("submission.csv", index=False)
print("submission.csv written.")