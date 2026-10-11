"""Plotting helpers shared by the notebook, the Streamlit app and the README assets."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

NAVY, TEAL, RED, SLATE, GRID = "#0F2A43", "#0E7C86", "#B3382C", "#5B6B7B", "#E2E7ED"
LABELS = {0: "Non-diabetic", 1: "Diabetic"}
PAL = {"Non-diabetic": TEAL, "Diabetic": RED}
LINE_COLORS = ["#0F2A43", "#0E7C86", "#B3382C", "#C27C0E", "#6C5B9E", "#2E7D5B", "#8C6D62", "#5B6B7B"]
NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
NICE = {"age": "Age (years)", "bmi": "BMI", "HbA1c_level": "HbA1c level (%)",
        "blood_glucose_level": "Blood glucose (mg/dL)"}


def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
        "grid.color": "#EEF1F5", "axes.labelcolor": NAVY, "text.color": NAVY,
        "xtick.color": SLATE, "ytick.color": SLATE, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlecolor": NAVY,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _labelled(df, target="diabetes", n=25000):
    d = df.sample(n, random_state=42) if len(df) > n else df
    return d.assign(**{target: d[target].map(LABELS)})


def class_distribution(df, target="diabetes"):
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values, color=[TEAL, RED], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + counts.max() * 0.015,
                f"{v:,}  ({v / counts.sum() * 100:.1f}%)", ha="center", fontsize=10)
    ax.set_title("Class distribution")
    ax.set_ylabel("Records")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


def numeric_distributions(df, target="diabetes"):
    d = _labelled(df, target)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for ax, f in zip(axes.ravel(), NUMERIC):
        sns.kdeplot(data=d, x=f, hue=target, fill=True, common_norm=False, alpha=0.35,
                    palette=PAL, linewidth=1.5, ax=ax, legend=(f == NUMERIC[0]))
        ax.set_title(NICE[f])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Feature distributions by outcome", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def boxplots_by_outcome(df, target="diabetes"):
    d = _labelled(df, target)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    for ax, f in zip(axes, NUMERIC):
        sns.boxplot(data=d, x=target, y=f, hue=target, palette=PAL, order=list(PAL), legend=False,
                    width=0.55, fliersize=2, linewidth=1, ax=ax)
        ax.set_title(NICE[f], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


def categorical_rates(df, target="diabetes"):
    cols = [("gender", "Gender"), ("smoking_history", "Smoking history"),
            ("hypertension", "Hypertension"), ("heart_disease", "Heart disease")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (c, title) in zip(axes.ravel(), cols):
        rate = (df.groupby(c)[target].mean() * 100).sort_values(ascending=False)
        labels = [{0: "No", 1: "Yes"}.get(i, str(i)) for i in rate.index]
        bars = ax.bar(labels, rate.values, color=TEAL, width=0.6)
        for b, v in zip(bars, rate.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
        ax.axhline(df[target].mean() * 100, color=RED, ls="--", lw=1)
        ax.set_title(f"Diabetes rate by {title.lower()}")
        ax.set_ylim(0, max(rate.max() * 1.2, 1))
        ax.tick_params(axis="x", labelsize=9)
    fig.text(0.99, 0.005, "Dashed line = overall rate", ha="right", fontsize=8, color=SLATE)
    fig.tight_layout()
    return fig


def correlation_heatmap(df):
    cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = LinearSegmentedColormap.from_list("tr", [TEAL, "white", RED])
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, linewidths=1,
                linecolor="white", cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 9})
    ax.set_title("Correlation matrix")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def accuracy_bar(results, baseline=None):
    r = results.sort_values("Accuracy")
    colors = [TEAL if m == results.index[0] else "#B8C4D0" for m in r.index]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.barh(r.index, r["Accuracy"], color=colors, height=0.6)
    for b, v in zip(bars, r["Accuracy"]):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v * 100:.2f}%", va="center", fontsize=9)
    lo = (baseline - 0.02) if baseline else r["Accuracy"].min() - 0.03
    ax.set_xlim(lo, min(1.0, r["Accuracy"].max() + 0.02))
    if baseline:
        ax.axvline(baseline, color=RED, ls="--", lw=1.2)
        ax.text(baseline + 0.0005, -0.75, f"majority-class baseline {baseline * 100:.1f}%",
                color=RED, fontsize=8, va="center")
    ax.set_xlabel("Test accuracy")
    ax.set_title("Model accuracy (best highlighted)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def metrics_comparison(results):
    cols = [c for c in ["Accuracy", "F1 Score", "ROC-AUC", "Diabetic Recall"] if c in results.columns]
    r = results[cols]
    x = np.arange(len(r))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for i, (c, col) in enumerate(zip(cols, [NAVY, TEAL, "#C27C0E", RED])):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, r[c], w, label=c, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(r.index, rotation=20, ha="right")
    ax.set_ylim(max(0, r.values.min() - 0.08), 1.0)
    ax.set_title("Model comparison across metrics", pad=30)
    ax.legend(frameon=False, ncol=len(cols), loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout()
    return fig


def roc_curves(scores, y_test):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for (name, s), col in zip(scores.items(), LINE_COLORS):
        fpr, tpr, _ = roc_curve(y_test, s)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=SLATE, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def pr_curves(scores, y_test):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for (name, s), col in zip(scores.items(), LINE_COLORS):
        p, r, _ = precision_recall_curve(y_test, s)
        ax.plot(r, p, color=col, lw=1.8, label=f"{name} ({average_precision_score(y_test, s):.3f})")
    ax.axhline(float(np.mean(y_test)), ls="--", color=SLATE, lw=1)
    ax.set_xlabel("Recall (diabetic)")
    ax.set_ylabel("Precision (diabetic)")
    ax.set_title("Precision-recall curves")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def confusion_matrix_fig(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])
    cmap = LinearSegmentedColormap.from_list("t", ["white", TEAL])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, cbar=False, linewidths=1, linecolor="white",
                xticklabels=[LABELS[0], LABELS[1]], yticklabels=[LABELS[0], LABELS[1]],
                annot_kws={"size": 12}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (row % shown)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def importance_fig(importances, title="Feature importance (permutation)", xlabel="Drop in ROC-AUC when shuffled"):
    s = importances.sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(s.index, s.values, color=TEAL, height=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def patient_position(df, patient, target="diabetes"):
    """Where a patient's HbA1c and glucose fall relative to the two classes."""
    d = _labelled(df, target)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, f in zip(axes, ["HbA1c_level", "blood_glucose_level"]):
        sns.kdeplot(data=d, x=f, hue=target, fill=True, common_norm=False, alpha=0.3,
                    palette=PAL, linewidth=1.3, ax=ax, legend=(f == "HbA1c_level"))
        ax.axvline(patient[f], color=NAVY, lw=2)
        ax.text(patient[f], ax.get_ylim()[1] * 0.97, "  patient", color=NAVY, fontsize=9, va="top")
        ax.set_title(NICE[f], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    return fig
