import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load clean dataset
df = pd.read_csv("Project_Dataset_Clean.csv")

# Encode labels
le = LabelEncoder()
df["LabelEncoded"] = le.fit_transform(df["Label"])

# Features and target
X = df[["Voltage"]]
y = df["LabelEncoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "voltage_classifier_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model retrained and saved successfully!")