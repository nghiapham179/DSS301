"""
train_model.py — Drone DSS | DSS301 Course
===========================================
Python 3.14 compatible — no joblib, no n_jobs, no cross_val_score.
CV duoc viet bang vong for loop thuan tuy.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ── CONFIG ──────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "drone_data_clean.csv"
MODEL_DIR = BASE_DIR / "Model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "battery_level", "flight_time", "signal_strength",
    "temperature", "wind_speed", "gps_accuracy",
    "altitude", "speed", "humidity", "pressure",
]

TARGETS = [
    ("operation_risk",     "operation_risk_model"),
    ("maintenance_action", "maintenance_action_model"),
    ("recommendation",     "recommendation_model"),
]

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
)

CV_FOLDS  = 5
TEST_SIZE = 0.2


# ── LOGGING ─────────────────────────────────────────────────────────────────

LINE  = "-" * 62
DLINE = "=" * 62


def _bar(pct, width=26):
    filled = int(width * max(0.0, min(1.0, float(pct))))
    return "[" + "#" * filled + "." * (width - filled) + f"] {float(pct)*100:5.1f}%"


def log(msg=""):
    print(msg, flush=True)


def log_header(title):
    log(f"\n{DLINE}\n  {title}\n{DLINE}")


def log_section(title):
    log(f"\n{LINE}\n  {title}\n{LINE}")


# ── DATA ─────────────────────────────────────────────────────────────────────

def load_data():
    log_section("Loading & Validating Data")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    log(f"  Rows    : {len(df):,}")
    log(f"  Columns : {df.shape[1]}")

    missing_feat   = [c for c in FEATURES if c not in df.columns]
    missing_target = [t for t, _ in TARGETS if t not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing feature columns: {missing_feat}")
    if missing_target:
        raise ValueError(f"Missing target columns: {missing_target}")
    log("  All required columns : OK")

    nan_counts = df[FEATURES].isnull().sum()
    if nan_counts.any():
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
        log("  NaN filled with median")
    else:
        log("  No missing values : OK")

    log(f"\n  {'Feature':<22} {'Min':>9} {'Median':>9} {'Max':>9}")
    log(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9}")
    for feat in FEATURES:
        log(f"  {feat:<22} {df[feat].min():>9.2f} {df[feat].median():>9.2f} {df[feat].max():>9.2f}")

    return df


# ── TRAIN ONE MODEL ──────────────────────────────────────────────────────────

def train_one(df, target_col, model_name, X, all_importances):
    log_section(f"Target: {target_col}  |  Model: {model_name}")

    le = LabelEncoder()
    y  = le.fit_transform(df[target_col])
    classes = le.classes_
    log(f"  Classes ({len(classes)}): {list(classes)}")

    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        log(f"    {classes[u]:<35} {c:>7,}  ({c/len(y)*100:5.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    log(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── Cross-validation: pure Python for loop, zero joblib ──────────────
    log(f"\n  Running {CV_FOLDS}-fold CV (no joblib, Python 3.14 safe) ...")
    skf           = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    X_arr         = np.array(X_train)
    y_arr         = np.array(y_train)
    cv_fold_scores = []
    t0            = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        fold_clf = RandomForestClassifier(**RF_PARAMS)
        fold_clf.fit(X_tr, y_tr)
        fold_acc = accuracy_score(y_va, fold_clf.predict(X_va))
        cv_fold_scores.append(fold_acc)
        log(f"    Fold {fold_idx} : {_bar(fold_acc)}  {fold_acc:.4f}")

    cv_scores = np.array(cv_fold_scores)
    cv_time   = time.time() - t0
    log(f"  CV done in {cv_time:.1f}s")
    log(f"  CV mean : {cv_scores.mean():.4f}  std : {cv_scores.std():.4f}")

    # ── Train final model ─────────────────────────────────────────────────
    log(f"\n  Training final model ...")
    t0    = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(np.array(X_train), y_train)
    train_time = time.time() - t0
    log(f"  Done in {train_time:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred      = model.predict(np.array(X_test))
    acc         = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, target_names=classes, zero_division=0, output_dict=True
    )
    report_str  = classification_report(
        y_test, y_pred, target_names=classes, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    log(f"\n  Test accuracy : {acc:.4f}  {_bar(acc)}")
    log("\n  Classification Report:")
    for line in report_str.split("\n"):
        if line.strip():
            log(f"    {line}")

    cw = max(len(c) for c in classes) + 2
    log("\n  Confusion Matrix:")
    log("  " + " " * cw + "".join(f"{c:>{cw}}" for c in classes))
    for i, row in enumerate(cm):
        log("  " + f"{classes[i]:<{cw}}" + "".join(f"{v:>{cw}}" for v in row))

    # ── Feature importance ────────────────────────────────────────────────
    importances = model.feature_importances_
    all_importances.append(importances)
    log("\n  Top 5 features:")
    for rank, idx in enumerate(np.argsort(importances)[::-1][:5], 1):
        log(f"    {rank}. {FEATURES[idx]:<20} {_bar(importances[idx], 20)}  {importances[idx]:.4f}")

    # ── Save files ────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
    joblib.dump(le,    MODEL_DIR / f"{model_name}_label_encoder.joblib")
    log(f"\n  Saved : {model_name}.joblib")
    log(f"  Saved : {model_name}_label_encoder.joblib")

    metrics = {
        "target":              target_col,
        "model_name":          model_name,
        "n_estimators":        RF_PARAMS["n_estimators"],
        "test_size":           TEST_SIZE,
        "cv_folds":            CV_FOLDS,
        "train_samples":       int(len(X_train)),
        "test_samples":        int(len(X_test)),
        "accuracy":            round(float(acc), 4),
        "cv_mean":             round(float(cv_scores.mean()), 4),
        "cv_std":              round(float(cv_scores.std()), 4),
        "classes":             list(classes),
        "per_class_f1":        {c: round(report_dict[c]["f1-score"],  4) for c in classes if c in report_dict},
        "per_class_precision": {c: round(report_dict[c]["precision"], 4) for c in classes if c in report_dict},
        "per_class_recall":    {c: round(report_dict[c]["recall"],    4) for c in classes if c in report_dict},
        "confusion_matrix":    cm.tolist(),
        "train_time_s":        round(train_time, 2),
        "trained_at":          pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(MODEL_DIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log(f"  Saved : {model_name}_metrics.json")

    return metrics


# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────

def save_feature_importance(all_importances):
    avg   = np.mean(all_importances, axis=0)
    fi_df = (
        pd.DataFrame({"feature": FEATURES, "importance": avg})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["rank"] = range(1, len(fi_df) + 1)
    fi_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    log("\n  Averaged feature importance (3 models):")
    for _, row in fi_df.iterrows():
        log(f"  {int(row['rank']):>2}. {row['feature']:<22} {_bar(row['importance'], 20)}  {row['importance']:.4f}")
    log("  Saved : feature_importance.csv")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    log_header("DRONE DSS — MODEL TRAINING | DSS301")
    log(f"  n_estimators : {RF_PARAMS['n_estimators']}")
    log(f"  CV folds     : {CV_FOLDS}")
    log(f"  Test size    : {TEST_SIZE:.0%}")
    log(f"  class_weight : {RF_PARAMS['class_weight']}")
    log(f"  n_jobs       : NOT USED (Python 3.14 safe)")

    wall_start = time.time()

    df = load_data()
    X  = df[FEATURES]

    log_header("Training 3 Models")
    all_importances = []
    summary         = []

    for target_col, model_name in TARGETS:
        m = train_one(df, target_col, model_name, X, all_importances)
        summary.append(m)

    log_section("Feature Importance Export")
    save_feature_importance(all_importances)

    total_time = time.time() - wall_start
    log_header("Training Complete")
    log(f"  {'Target':<26} {'Accuracy':>9} {'CV Mean':>9} {'CV +/-':>8}")
    log(f"  {'-'*26} {'-'*9} {'-'*9} {'-'*8}")
    for m in summary:
        log(f"  {m['target']:<26} {m['accuracy']:>9.4f} {m['cv_mean']:>9.4f} {m['cv_std']:>8.4f}")

    log(f"\n  Total time : {total_time:.1f}s")
    log(f"\n  Files in Model/:")
    for p in sorted(MODEL_DIR.iterdir()):
        log(f"    {p.name:<50} {p.stat().st_size/1024:>7.1f} KB")
    log(f"\n{DLINE}\n")


if __name__ == "__main__":
    main()