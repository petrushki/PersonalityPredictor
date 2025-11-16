import pickle 
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

model_file = 'model.bin'

app = FastAPI(title = "personality-prediction")

def load_model(filename):
    with open(filename) as f_in:
        model = pickle.load(f_in)
    return model

def predict_single(sample):
    model = load_model(model_file)
    y_pred = model.predict_proba(sample)[0,1]
    return y_pred

@app.post("/predict")
def predict(sample: Dict[str, Any]):
    y_pred = predict_single(sample)
    return {
        'probability': float(y_pred),
        'introvert': bool(y_pred > 0.5),
        'extrovert': bool(y_pred <= 0.5)
    }
        
    