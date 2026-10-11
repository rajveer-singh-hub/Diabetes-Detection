"""Train every model, pick the best one, and save artifacts for the apps.

Run once:  python train.py
"""
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from model import ARTIFACT_DIR, FEATURES, TARGET, build_models, load_clean, positive_scores


def main():
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    models, rows, scores = build_models(), [], {}
    for name, m in models.items():
        print(f"Training {name} ...", flush=True)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        scores[name] = positive_scores(m, X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, scores[name]),
            "Diabetic Recall": recall_score(y_test, pred, pos_label=1),
            "Diabetic F1": f1_score(y_test, pred, pos_label=1),
        })
    results = (pd.DataFrame(rows).sort_values(["Accuracy", "F1 Score"], ascending=False)
               .set_index("Model"))
    best_name = results.index[0]
    best = models[best_name]
    best_pred = best.predict(X_test)
    print(results.round(4), "\nBest model:", best_name)

    imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=42,
                                 scoring="roc_auc", n_jobs=1)
    importances = pd.Series(imp.importances_mean, index=FEATURES)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": best, "name": best_name}, ARTIFACT_DIR / "best_model.joblib", compress=3)
    joblib.dump({
        "results": results, "scores": scores, "y_test": y_test.to_numpy(),
        "best_name": best_name, "best_pred": best_pred, "importances": importances,
        "info": {"rows": len(df), "train": len(X_train), "test": len(X_test),
                 "baseline": 1 - float(y_test.mean())},
    }, ARTIFACT_DIR / "summary.joblib", compress=3)
    print("Saved artifacts to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
