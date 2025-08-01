# predict.py

import sys
import json
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load input
if len(sys.argv) > 1:
    input_json = sys.argv[1]
else:
    input_json = sys.stdin.read()

input_dict = json.loads(input_json)
input_array = np.array([list(input_dict.values())])
input_scaled = scaler.transform(input_array)

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

result = {
    "prediction": int(prediction),
    "probability": round(probability, 4)
}

print(json.dumps(result))
