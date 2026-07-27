import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
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

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ----------------------------------
# New Employee Details
# ----------------------------------

new_employee = [[
    28,     # Age
    3500,   # Monthly Income
    12,     # Distance From Home
    1,      # Job Satisfaction
    2,      # Work Life Balance
    2,      # Years At Company
    1,      # OverTime (Yes)
    1       # Gender (Male)
]]

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("⚠ High Risk of Attrition")
else:
    print("✅ Low Risk of Attrition")