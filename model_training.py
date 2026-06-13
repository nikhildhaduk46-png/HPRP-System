import pandas as pd
import joblib, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier

os.makedirs(MODEL_DIR, exist_ok=True)

def train(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "LogisticRegression" : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree"       : DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "RandomForest"       : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting"   : GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    trained = {}
    print("\n📊 Cross-Validation AUC Scores (5-fold):")
    for name, model in models.items():
        cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='roc_auc')
        model.fit(X_train_sc, y_train)
        trained[name] = model
        print(f"   {name:25s}  AUC: {cv.mean():.4f} ± {cv.std():.4f}")
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    joblib.dump(scaler,          os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

    print("\n✅ All models + scaler saved to models\\")
    return X_test_sc, y_test, trained, X.columns.tolist()

