import joblib
import pandas as pd
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_DIR

def predict_patient(patient_dict: dict, model_name: str = "RandomForest"):
    """
    Pass a patient dictionary and get a risk prediction.
    model_name options: RandomForest | GradientBoosting |
                        DecisionTree | LogisticRegression
    """
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model         = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    df     = pd.DataFrame([patient_dict])[feature_names]
    scaled = scaler.transform(df)

    pred  = model.predict(scaled)[0]
    prob  = model.predict_proba(scaled)[0][1]
    label = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"

    print(f"\n  Prediction  : {label}")
    print(f"  Probability : {prob:.2%}")
    return {"prediction": int(pred), "probability": round(float(prob), 4), "label": label}