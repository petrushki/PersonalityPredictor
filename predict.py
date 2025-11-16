import pickle 
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import numpy as np

FEATURES = [
    "time_spent_alone",
    "stage_fear",
    "social_event_attendance",
    "going_outside",
    "drained_after_socializing",
    "friends_circle_size",
    "post_frequency"
]

model_file = 'model.bin'

app = FastAPI(title = "personality-prediction")

def preprocess_sample(sample: Dict[str, Any]):
    """Convert dict input → numpy array in correct feature order."""
    values = []

    for feature in FEATURES:
        if feature not in sample:
            raise ValueError(f"Missing required field: {feature}")
        values.append(sample[feature])

    X = np.array(values, dtype=float).reshape(1, -1)
    return X


def load_model(filename):
    with open(filename, 'rb') as f_in:
        model = pickle.load(f_in)
    return model

def predict_single(sample):
    model = load_model(model_file)
    X = preprocess_sample(sample)
    y_pred = model.predict_proba(X)[0,1]
    return y_pred

@app.post("/predict")
def predict(sample: Dict[str, Any]):
    y_pred = predict_single(sample)
    return {
        'probability': float(y_pred),
        'introvert': bool(y_pred > 0.5),
        'extrovert': bool(y_pred <= 0.5)
    }
        
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)