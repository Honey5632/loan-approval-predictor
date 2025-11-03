from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('loan_model.pkl')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = int(request.form["gender"])
        married = int(request.form["married"])
        income = float(request.form["income"])
        loan_amount = float(request.form["loan"])
        credit_history = int(request.form["credit"])

        features = np.array([[gender, married, income, loan_amount, credit_history]])
        prediction = model.predict(features)[0]

        result = "Approved" if prediction == 1 else "Rejected"

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
