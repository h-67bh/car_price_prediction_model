import pickle
import json

# Load trained pipeline
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

preprocessor = model.named_steps["columntransformer"]
reg = model.named_steps["linearregression"]

# Get final feature names after encoding
feature_names = preprocessor.get_feature_names_out()

data = {
    "intercept": float(reg.intercept_),
    "coefficients": reg.coef_.tolist(),
    "feature_names": feature_names.tolist()
}

with open("model_full_export.json", "w") as f:
    json.dump(data, f)

print("Saved model_full_export.json")