# Import required libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import re
import os

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

# Load dataset (used for dropdown values only)
car_data = pd.read_csv("Cleaned_Car_data.csv")


# -------------------------
# Home Page Route
# -------------------------
@app.route("/")
def index():
    """
    This function renders the homepage
    and sends dropdown values to HTML.
    """

    companies = sorted(car_data["company"].unique())
    models = sorted(car_data["name"].unique())
    years = sorted(car_data["year"].unique(), reverse=True)
    fuel_types = sorted(car_data["fuel_type"].unique())

    return render_template(
        "index.html",
        companies=companies,
        models=models,
        years=years,
        fuel_types=fuel_types
    )


# -------------------------
# Prediction Route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")
    car_model = request.form.get("car_model")
    year = int(request.form.get("year"))
    fuel_type = request.form.get("fuel_type")
    kms_driven = request.form.get("kms_driven")

    # Clean kms (optional safety)
    kms_driven_int = int(re.sub(r"[^0-9]", "", str(kms_driven))) if kms_driven else 0

    input_df = pd.DataFrame([{
        "name": car_model,
        "company": company,
        "year": year,
        "kms_driven": kms_driven_int,
        "fuel_type": fuel_type
    }])

    prediction = round(model.predict(input_df)[0], 2)

    # Dropdown values
    companies = sorted(car_data["company"].dropna().unique())
    models = sorted(car_data["name"].dropna().unique())
    years = sorted(car_data["year"].dropna().unique(), reverse=True)
    fuel_types = sorted(car_data["fuel_type"].dropna().unique())

    return render_template(
        "index.html",
        companies=companies,
        models=models,
        years=years,
        fuel_types=fuel_types,
        prediction_text=f"Estimated Price: ₹ {prediction}",
        selected_company=company,
        selected_model=car_model,
        selected_year=year,
        selected_fuel=fuel_type,
        selected_kms=kms_driven
    )


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
