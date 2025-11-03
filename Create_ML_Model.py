# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Example dataset — replace with a real dataset (CSV) later
data = {
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female'],
    'Married': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'ApplicantIncome': [5000, 3000, 7000, 2000, 4000, 2500, 6000, 3500],
    'LoanAmount': [150, 100, 200, 80, 120, 60, 180, 110],
    'Credit_History': [1, 0, 1, 0, 1, 1, 1, 0],
    'Loan_Status': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
}
df = pd.DataFrame(data)

# Simple encoding for demo (you can use sklearn Pipelines later)
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df['Married'] = df['Married'].map({'Yes':1, 'No':0})
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")
print("Model trained and saved: loan_model.pkl")
