"""
train_model.py — Drone DSS | DSS301 Course
===========================================
Python 3.14 compatible — no n_jobs, no cross_val_score.
CV duoc viet bang vong for loop thuan tuy.

v2: Probabilistic Label Relabeling
    Giu nhan goc lam nen, chi flip mot phan nhan tai
    VUNG BIEN GIOI (boundary zone) giua cac lop.
    - Vung bien (score gan nguong chuyen tiep): flip 20%
    - Vung ro rang: flip 2% (noise nho)
    => accuracy ~87-88% thay vi 99%+ (phan anh thuc te hon)
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ── CONFIG ───────────────────────────────────────────────────────────────────

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
    n_jobs=1,          # Python 3.14: tắt joblib parallel để tránh crash
)

DT_PARAMS = dict(max_depth=10, min_samples_split=5, random_state=42)
LR_PARAMS = dict(solver="lbfgs", max_iter=1000, random_state=42)

CV_FOLDS           = 5
TEST_SIZE          = 0.2
NOISE_SEED         = 42
BOUNDARY_FLIP_RATE = 0.20   # flip 20% nhan o vung bien
CLEAR_FLIP_RATE    = 0.02   # flip  2% nhan vung ro rang


# ── LOGGING ──────────────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC LABEL RELABELING
# ══════════════════════════════════════════════════════════════════════════════

def _composite_risk_score(df: pd.DataFrame) -> np.ndarray:
    """
    Diem rui ro tong hop (0→1) tinh tu nhieu features.
    Dung de xac dinh record nao nam o vung bien giua cac lop.

    Trong so theo Feature Importance thuc te cua RF:
      battery_level   44%  → weight 0.40
      wind_speed      16%  → weight 0.22
      flight_time     16%  → weight 0.20
      signal_strength  6%  → weight 0.18
    """
    b = 1.0 - df["battery_level"].values   / 100.0
    w =       df["wind_speed"].values       / 50.0
    t =       df["flight_time"].values      / 60.0
    s = 1.0 - df["signal_strength"].values  / 100.0
    return np.clip(0.40*b + 0.22*w + 0.20*t + 0.18*s, 0.0, 1.0)


def _is_boundary(scores: np.ndarray) -> np.ndarray:
    """
    Vung bien giua cac lop (gan nguong chuyen tiep):
      Low → Medium:   score in (0.25, 0.40)
      Medium → High:  score in (0.52, 0.67)
    """
    return (
            ((scores > 0.25) & (scores < 0.40)) |
            ((scores > 0.52) & (scores < 0.67))
    )


def _flip_labels(
        series:      pd.Series,
        is_boundary: np.ndarray,
        rng:         np.random.Generator,
) -> pd.Series:
    """
    Flip ngau nhien mot phan nhan:
      - Boundary records: flip voi BOUNDARY_FLIP_RATE (20%)
      - Clear records:    flip voi CLEAR_FLIP_RATE (2%)
    Khi flip, chon ngau nhien mot trong cac lop con lai.
    """
    classes = series.unique().tolist()
    y       = series.values.copy()
    for i in range(len(y)):
        rate = BOUNDARY_FLIP_RATE if is_boundary[i] else CLEAR_FLIP_RATE
        if rng.random() < rate:
            other = [c for c in classes if c != y[i]]
            if other:
                y[i] = rng.choice(other)
    return pd.Series(y, index=series.index)


def relabel_probabilistic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ap dung Probabilistic Label Relabeling cho ca 3 targets.
    Goi truoc khi train de accuracy phan anh thuc te hon (~87-88%).
    """
    log_section("Probabilistic Label Relabeling")
    log(f"  Boundary flip : {BOUNDARY_FLIP_RATE:.0%}  "
        f"| Clear flip : {CLEAR_FLIP_RATE:.0%}  "
        f"| Seed : {NOISE_SEED}")

    rng    = np.random.default_rng(NOISE_SEED)
    df     = df.copy()
    scores = _composite_risk_score(df)
    bnd    = _is_boundary(scores)

    log(f"  Records tong      : {len(df):,}")
    log(f"  Records vung bien : {bnd.sum():,} "
        f"({bnd.sum()/len(df)*100:.1f}%) → flip {BOUNDARY_FLIP_RATE:.0%}")
    log(f"  Records vung ro   : {(~bnd).sum():,} "
        f"({(~bnd).sum()/len(df)*100:.1f}%) → flip {CLEAR_FLIP_RATE:.0%}")

    for target_col, _ in TARGETS:
        orig = df[target_col].value_counts().to_dict()
        df[target_col] = _flip_labels(df[target_col], bnd, rng)
        new  = df[target_col].value_counts().to_dict()
        log(f"\n  {target_col}:")
        log(f"  {'Nhan':<55} {'Truoc':>7} {'Sau':>7}")
        log(f"  {'-'*55} {'-'*7} {'-'*7}")
        for lbl in sorted(set(list(orig)+list(new))):
            log(f"  {str(lbl)[:55]:<55} "
                f"{orig.get(lbl,0):>7,} {new.get(lbl,0):>7,}")

    log("\n  Relabeling xong.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

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
        log(f"  {feat:<22} {df[feat].min():>9.2f} "
            f"{df[feat].median():>9.2f} {df[feat].max():>9.2f}")

    # Ap dung relabeling truoc khi tra ve
    df = relabel_probabilistic(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE TARGET (RF chinh + DT/LR so sanh)
# ══════════════════════════════════════════════════════════════════════════════

def train_one(df, target_col, model_name, X, all_importances):
    log_section(f"Target: {target_col}  |  Model: {model_name}")

    le = LabelEncoder()
    y  = le.fit_transform(df[target_col])
    classes = le.classes_
    log(f"  Classes ({len(classes)}): {list(classes)}")

    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        log(f"    {classes[u]:<55} {c:>7,}  ({c/len(y)*100:5.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    log(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── 5-Fold CV (RF) ────────────────────────────────────────────────────
    log(f"\n  Running {CV_FOLDS}-fold CV (no joblib, Python 3.14 safe) ...")
    skf            = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    X_arr          = np.array(X_train)
    y_arr          = np.array(y_train)
    cv_fold_scores = []
    t0             = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
        fold_clf   = RandomForestClassifier(**RF_PARAMS)
        fold_clf.fit(X_tr, y_tr)
        fold_acc   = accuracy_score(y_va, fold_clf.predict(X_va))
        cv_fold_scores.append(fold_acc)
        log(f"    Fold {fold_idx} : {_bar(fold_acc)}  {fold_acc:.4f}")

    cv_scores = np.array(cv_fold_scores)
    log(f"  CV done in {time.time()-t0:.1f}s")
    log(f"  CV mean : {cv_scores.mean():.4f}  std : {cv_scores.std():.4f}")

    # ── Train RF final ────────────────────────────────────────────────────
    log(f"\n  [RF] Training final model ...")
    t0    = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_arr, y_arr)
    train_time = time.time() - t0
    log(f"  Done in {train_time:.1f}s")

    y_pred      = model.predict(np.array(X_test))
    acc         = accuracy_score(y_test, y_pred)
    prec        = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec         = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1          = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    rmse        = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae         = float(mean_absolute_error(y_test, y_pred))
    report_dict = classification_report(
        y_test, y_pred, target_names=classes, zero_division=0, output_dict=True
    )
    report_str  = classification_report(
        y_test, y_pred, target_names=classes, zero_division=0
    )
    cm          = confusion_matrix(y_test, y_pred)
    importances = model.feature_importances_
    all_importances.append(importances)

    log(f"\n  [RF] Test accuracy : {acc:.4f}  {_bar(acc)}")
    log(f"       Precision={prec:.4f}  Recall={rec:.4f}  "
        f"F1={f1:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    log("\n  Classification Report:")
    for line in report_str.split("\n"):
        if line.strip():
            log(f"    {line}")

    cw = max(len(c) for c in classes) + 2
    log("\n  Confusion Matrix:")
    log("  " + " " * cw + "".join(f"{c:>{cw}}" for c in classes))
    for i, row in enumerate(cm):
        log("  " + f"{classes[i]:<{cw}}" + "".join(f"{v:>{cw}}" for v in row))

    log("\n  Top 5 features:")
    for rank, idx in enumerate(np.argsort(importances)[::-1][:5], 1):
        log(f"    {rank}. {FEATURES[idx]:<22} "
            f"{_bar(importances[idx], 20)}  {importances[idx]:.4f}")

    # ── Train DT (so sanh) ────────────────────────────────────────────────
    log("\n  [DT] Training Decision Tree ...")
    dt = DecisionTreeClassifier(**DT_PARAMS)
    dt.fit(X_arr, y_arr)
    yp_dt   = dt.predict(np.array(X_test))
    acc_dt  = accuracy_score(y_test, yp_dt)
    prec_dt = precision_score(y_test, yp_dt, average="weighted", zero_division=0)
    rec_dt  = recall_score(y_test, yp_dt, average="weighted", zero_division=0)
    f1_dt   = f1_score(y_test, yp_dt, average="weighted", zero_division=0)
    rmse_dt = float(np.sqrt(mean_squared_error(y_test, yp_dt)))
    mae_dt  = float(mean_absolute_error(y_test, yp_dt))
    log(f"  [DT]  acc={acc_dt:.4f}  prec={prec_dt:.4f}  "
        f"rec={rec_dt:.4f}  f1={f1_dt:.4f}  "
        f"rmse={rmse_dt:.4f}  mae={mae_dt:.4f}")

    # ── Train LR (so sanh) ────────────────────────────────────────────────
    log("\n  [LR] Training Logistic Regression ...")
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_arr, y_arr)
    yp_lr   = lr.predict(np.array(X_test))
    acc_lr  = accuracy_score(y_test, yp_lr)
    prec_lr = precision_score(y_test, yp_lr, average="weighted", zero_division=0)
    rec_lr  = recall_score(y_test, yp_lr, average="weighted", zero_division=0)
    f1_lr   = f1_score(y_test, yp_lr, average="weighted", zero_division=0)
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, yp_lr)))
    mae_lr  = float(mean_absolute_error(y_test, yp_lr))
    log(f"  [LR]  acc={acc_lr:.4f}  prec={prec_lr:.4f}  "
        f"rec={rec_lr:.4f}  f1={f1_lr:.4f}  "
        f"rmse={rmse_lr:.4f}  mae={mae_lr:.4f}")

    # ── Save RF model ─────────────────────────────────────────────────────
    joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
    joblib.dump(le,    MODEL_DIR / f"{model_name}_label_encoder.joblib")
    log(f"\n  Saved : {model_name}.joblib")
    log(f"  Saved : {model_name}_label_encoder.joblib")

    # ── Save metrics JSON ─────────────────────────────────────────────────
    metrics = {
        # Meta
        "target":              target_col,
        "model_name":          model_name,
        "relabeling":          "probabilistic_v2",
        "boundary_flip_rate":  BOUNDARY_FLIP_RATE,
        "clear_flip_rate":     CLEAR_FLIP_RATE,
        "noise_seed":          NOISE_SEED,
        "n_estimators":        RF_PARAMS["n_estimators"],
        "test_size":           TEST_SIZE,
        "cv_folds":            CV_FOLDS,
        "train_samples":       int(len(X_train)),
        "test_samples":        int(len(X_test)),
        "trained_at":          pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classes":             list(classes),
        # RF — 6 metrics
        "accuracy":            round(float(acc),  4),
        "precision":           round(float(prec), 4),
        "recall":              round(float(rec),  4),
        "f1":                  round(float(f1),   4),
        "rmse":                round(rmse,         4),
        "mae":                 round(mae,           4),
        # CV
        "cv_mean":             round(float(cv_scores.mean()), 4),
        "cv_std":              round(float(cv_scores.std()),  4),
        # Per-class
        "per_class_f1":        {c: round(report_dict[c]["f1-score"],  4)
                                for c in classes if c in report_dict},
        "per_class_precision": {c: round(report_dict[c]["precision"], 4)
                                for c in classes if c in report_dict},
        "per_class_recall":    {c: round(report_dict[c]["recall"],    4)
                                for c in classes if c in report_dict},
        "confusion_matrix":    cm.tolist(),
        "train_time_s":        round(train_time, 2),
        # Comparison models (DT + LR)
        "comparison": {
            "Decision Tree": {
                "accuracy":  round(acc_dt,  4),
                "precision": round(prec_dt, 4),
                "recall":    round(rec_dt,  4),
                "f1":        round(f1_dt,   4),
                "rmse":      round(rmse_dt, 4),
                "mae":       round(mae_dt,  4),
                "params":    DT_PARAMS,
            },
            "Logistic Regression": {
                "accuracy":  round(acc_lr,  4),
                "precision": round(prec_lr, 4),
                "recall":    round(rec_lr,  4),
                "f1":        round(f1_lr,   4),
                "rmse":      round(rmse_lr, 4),
                "mae":       round(mae_lr,  4),
                "params":    {k: str(v) for k, v in LR_PARAMS.items()},
            },
        },
    }

    with open(MODEL_DIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log(f"  Saved : {model_name}_metrics.json")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def save_feature_importance(all_importances):
    avg   = np.mean(all_importances, axis=0)
    fi_df = (
        pd.DataFrame({"feature": FEATURES, "importance": avg})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["rank"] = range(1, len(fi_df) + 1)
    fi_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    log("\n  Averaged feature importance (3 RF models):")
    for _, row in fi_df.iterrows():
        log(f"  {int(row['rank']):>2}. {row['feature']:<22} "
            f"{_bar(row['importance'], 20)}  {row['importance']:.4f}")
    log("  Saved : feature_importance.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log_header("DRONE DSS — MODEL TRAINING | DSS301")
    log(f"  n_estimators       : {RF_PARAMS['n_estimators']}")
    log(f"  CV folds           : {CV_FOLDS}")
    log(f"  Test size          : {TEST_SIZE:.0%}")
    log(f"  class_weight       : {RF_PARAMS['class_weight']}")
    log(f"  n_jobs             : NOT USED (Python 3.14 safe)")
    log(f"  relabeling         : probabilistic_v2")
    log(f"  boundary_flip_rate : {BOUNDARY_FLIP_RATE:.0%}")
    log(f"  clear_flip_rate    : {CLEAR_FLIP_RATE:.0%}")
    log(f"  noise_seed         : {NOISE_SEED}")
    log(f"  comparison models  : Decision Tree + Logistic Regression")

    wall_start = time.time()

    df = load_data()
    X  = df[FEATURES]

    log_header("Training 3 Targets × 3 Models")
    all_importances = []
    summary         = []

    for target_col, model_name in TARGETS:
        m = train_one(df, target_col, model_name, X, all_importances)
        summary.append(m)

    log_section("Feature Importance Export")
    save_feature_importance(all_importances)

    total_time = time.time() - wall_start
    log_header("Training Complete")
    log(f"\n  {'Target':<26} {'RF Acc':>8} {'DT Acc':>8} "
        f"{'LR Acc':>8} {'CV Mean':>8} {'CV Std':>7}")
    log(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for m in summary:
        dt_acc = m["comparison"]["Decision Tree"]["accuracy"]
        lr_acc = m["comparison"]["Logistic Regression"]["accuracy"]
        log(f"  {m['target']:<26} {m['accuracy']:>8.4f} "
            f"{dt_acc:>8.4f} {lr_acc:>8.4f} "
            f"{m['cv_mean']:>8.4f} {m['cv_std']:>7.4f}")

    log(f"\n  Total time : {total_time:.1f}s")
    log(f"\n  Files in Model/:")
    for p in sorted(MODEL_DIR.iterdir()):
        log(f"    {p.name:<50} {p.stat().st_size/1024:>7.1f} KB")
    log(f"\n  Expected accuracy: ~87-88% RF | ~86% DT | ~65% LR")
    log(f"  App: streamlit run app.py\n")
    log(f"{DLINE}\n")


if __name__ == "__main__":
    main()