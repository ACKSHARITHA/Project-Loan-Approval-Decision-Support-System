import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib, json

# Generate synthetic dataset
np.random.seed(42)
n = 300
data = pd.DataFrame({
    "age": np.random.randint(20, 60, n),
    "income": np.random.randint(2000, 15000, n),
    "loan_amount": np.random.randint(500, 20000, n),
    "loan_term": np.random.choice([12, 24, 36, 48, 60], n),
    "credit_score": np.random.randint(300, 900, n),
    "employment": np.random.choice(["salaried", "self-employed", "unemployed"], n),
    "marital_status": np.random.choice(["single", "married", "divorced"], n),
    "education": np.random.choice(["undergrad", "graduate", "postgraduate"], n),
    "property_area": np.random.choice(["urban", "semiurban", "rural"], n),
})

# Target variable
data["approved"] = (
    (data["income"] > 4000) &
    (data["credit_score"] > 600) &
    (data["loan_amount"] < data["income"] * 5)
).astype(int)

# Save dataset
data.to_csv("loan_dataset.csv", index=False)

# Features and target
X = data.drop("approved", axis=1)
y = data["approved"]

categorical = X.select_dtypes(include="object").columns.tolist()
numerical = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numerical),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.joblib")

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Tree summary
clf = model.named_steps["clf"]
feature_names = model.named_steps["preprocess"].get_feature_names_out()
tree_text = export_text(clf, feature_names=list(feature_names))

# Save meta
with open("model_meta.json", "w") as f:
    json.dump({"accuracy": acc, "tree_text": tree_text}, f, indent=4)

print(f"Model trained. Accuracy: {acc:.2f}")
