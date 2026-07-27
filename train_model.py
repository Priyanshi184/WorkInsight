import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert categorical columns
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Features
X = df[
    [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "JobSatisfaction",
        "WorkLifeBalance",
        "YearsAtCompany",
        "OverTime",
        "Gender",
    ]
]

# Target
y = df["Attrition"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "models/attrition_model.pkl")

print("Model saved successfully!")