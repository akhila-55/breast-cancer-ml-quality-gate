import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("breast_cancer_v1_100.csv")

# Remove unnecessary unnamed/index columns if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Display dataset information
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# Assume the last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Convert categorical target if required
if y.dtype == "object":
    y = y.astype("category").cat.codes


# Convert categorical input columns if any
X = pd.get_dummies(X, drop_first=True)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")


# Quality gate
QUALITY_THRESHOLD = 0.90

if accuracy < QUALITY_THRESHOLD:
    raise ValueError(
        f"Quality Gate Failed: Accuracy {accuracy:.4f} "
        f"is below required threshold {QUALITY_THRESHOLD}"
    )

print("Quality Gate Passed!")


# Save model
joblib.dump(model, "breast_cancer_model.pkl")

print("Model saved successfully.")
