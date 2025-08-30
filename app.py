from flask import Flask, render_template, request
import joblib, json
import pandas as pd

app = Flask(__name__)

# Load model + metadata
model = joblib.load("model.joblib")
with open("model_meta.json", "r") as f:
    meta = json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = {k: v for k, v in request.form.items()}
    input_df = pd.DataFrame([form_data])

    # Convert numeric fields
    for col in ["age", "income", "loan_amount", "loan_term", "credit_score"]:
        input_df[col] = pd.to_numeric(input_df[col])

    # Prediction
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]

    return render_template("result.html",
                           form_data=form_data,
                           prediction="Approved" if pred == 1 else "Rejected",
                           probability=f"{proba:.2f}",
                           accuracy=f"{meta['accuracy']:.2f}",
                           tree_text=meta['tree_text'])

if __name__ == "__main__":
    app.run(debug=True)
