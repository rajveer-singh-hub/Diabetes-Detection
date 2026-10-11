"""Data loading, model definitions and prediction helpers (shared by every script)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "diabetes_prediction_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "diabetes"
NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_history"]
FEATURES = ["gender", "age", "hypertension", "heart_disease", "smoking_history",
            "bmi", "HbA1c_level", "blood_glucose_level"]
GENDERS = ["Female", "Male", "Other"]
SMOKING = ["never", "No Info", "former", "current", "not current", "ever"]


def load_clean():
    """Load the CSV and drop exact duplicate rows (they leak between train and test)."""
    df = pd.read_csv(DATA_PATH)
    return df.drop_duplicates().reset_index(drop=True)


def make_preprocessor():
    return ColumnTransformer(
        [("num", StandardScaler(), NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",  # hypertension / heart_disease are already 0/1
    )


def build_models():
    dt = lambda: DecisionTreeClassifier(random_state=42)
    lo = lambda: LogisticRegression(max_iter=1000)
    rf = lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    knn = lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    xgb = lambda: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                random_state=42, eval_metric="logloss", n_jobs=-1)
    svc = lambda: SVC(random_state=42)
    voting = VotingClassifier(
        [("dt", dt()), ("knn", knn()), ("lo", lo()), ("rf", rf()), ("xgb", xgb())], voting="soft")
    stacking = StackingClassifier(
        [("dt", dt()), ("rf", rf()), ("knn", knn()), ("xgb", xgb())],
        final_estimator=lo(), cv=3, n_jobs=1)
    clfs = {
        "Decision Tree": dt(), "Logistic Regression": lo(), "Random Forest": rf(), "KNN": knn(),
        "SVC": svc(), "XGBoost": xgb(), "Voting (soft)": voting, "Stacking": stacking,
    }
    return {n: Pipeline([("prep", make_preprocessor()), ("clf", c)]) for n, c in clfs.items()}


def positive_scores(model, X):
    """Probability of the positive class (decision score for models without probabilities)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def make_input(gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose):
    return pd.DataFrame([[gender, age, int(hypertension), int(heart_disease), smoking,
                          bmi, hba1c, glucose]], columns=FEATURES)


def risk_band(p):
    return "Low" if p < 0.20 else "Moderate" if p < 0.50 else "High"
