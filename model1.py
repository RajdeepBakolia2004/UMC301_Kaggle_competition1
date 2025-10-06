import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("/data/train.csv")
test  = pd.read_csv("/data/test.csv")

# Target & drop ID
y = train["song_popularity"]
X = train.drop(columns=["id", "song_popularity"])
X_test = test.drop(columns=["id"])
test_ids = test["id"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Validation Accuracy:",
      accuracy_score(y_valid, model.predict(X_valid)))

model.fit(X, y)
preds = model.predict(X_test)
submission = pd.DataFrame({
    "id": test_ids,
    "song_popularity": preds
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")