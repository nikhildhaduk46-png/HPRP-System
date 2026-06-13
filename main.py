"""
Hospital Patient Risk Prediction System
Run:  python main.py
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

from src.data_loader      import load_data, basic_info
from src.preprocessing    import preprocess
from src.model_training   import train
from src.model_evaluation import evaluate_all

def main():
    print("\n" + "="*60)
    print("  🏥 HOSPITAL PATIENT RISK PREDICTION SYSTEM")
    print("="*60)

    print("\n[1/4] Loading data ...")
    df = load_data()
    basic_info(df)

    print("\n[2/4] Preprocessing ...")
    df_clean = preprocess(df)

    print("\n[3/4] Training models ...")
    X_test, y_test, trained_models, feature_names = train(df_clean)

    print("\n[4/4] Evaluating models ...")
    evaluate_all(trained_models, X_test, y_test, feature_names)

    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLETE!")
    print("="*60)
    print("  Models  → models\\")
    print("  Reports → reports\\")
    print("  Web App → streamlit run app\\app.py")

if __name__ == "__main__":
    main()