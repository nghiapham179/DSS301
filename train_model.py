"""
train_model.py — Drone DSS | DSS301
====================================
Python 3.14 compatible — no n_jobs, no cross_val_score.
CV duoc viet bang vong for loop thuan tuy.

v3: Honest Evaluation Design (thay the v2 Probabilistic Relabeling)
    Nhan synthetic sinh tu luat => accuracy random-split cao la he qua
    tat yeu (model "hoc lai bo luat"). Thay vi tiem nhieu vao ca test
    de keo accuracy xuong (v2 — lam sai lech thuoc do), v3 giu nhan
    sach va tra loi 3 cau hoi nghien cuu:

    [1] GENERALIZATION — GroupKFold theo drone_id: model tong quat
        sang drone CHUA TUNG THAY tot den dau? So voi random CV de
        do "leakage gap".  (Kapoor & Narayanan 2023, Patterns;
        Roberts et al. 2021, Nature Machine Intelligence)

    [2] NOISE ROBUSTNESS — tiem nhieu nhan phu thuoc dac trung (NNAR,
        vung bien flip nhieu hon vung ro) vao TRAIN ONLY o cac muc
        0/5/10/20/30%, test giu SACH. So do ben nhieu RF vs DT vs LR.
        (Frenay & Verleysen 2014, IEEE TNNLS)

    [3] COST-SENSITIVE — ma tran chi phi bat doi xung (bo sot High
        dat gap 10 lan bao dong nham) + quy tac Bayes toi thieu chi
        phi ky vong thay cho argmax.  (Elkan 2001, IJCAI)

Outputs:
    Model/<name>.joblib, <name>_label_encoder.joblib, <name>_metrics.json
    Model/feature_importance.csv
    Model/noise_robustness.json   ← thi nghiem [2]
    Model/cost_sensitive.json     ← thi nghiem [3]
"""

import json
import sys
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
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     train_test_split)
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

# Target chinh cho 2 thi nghiem nghien cuu (noise + cost)
PRIMARY_TARGET = "operation_risk"

# ══════════════════════════════════════════════════════════════════════════════
# ANH XA drone_id → 3 dong DJI pho bien nhat Viet Nam
# (Mini 3 = pho thong, Air 3 = tam trung, Mavic 3 Pro = cao cap)
# Dung de hien thi per-drone breakdown theo ten model that.
# ══════════════════════════════════════════════════════════════════════════════
DRONE_NAME_MAP = {
    "Drone_1":  "DJI Mini 3",
    "Drone_4":  "DJI Mini 3",
    "Drone_7":  "DJI Mini 3",
    "Drone_10": "DJI Mini 3",
    "Drone_2":  "DJI Air 3",
    "Drone_5":  "DJI Air 3",
    "Drone_8":  "DJI Air 3",
    "Drone_3":  "DJI Mavic 3 Pro",
    "Drone_6":  "DJI Mavic 3 Pro",
    "Drone_9":  "DJI Mavic 3 Pro",
}

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

CV_FOLDS   = 5
TEST_SIZE  = 0.2
NOISE_SEED = 42

# ── Thi nghiem [2] NOISE ROBUSTNESS ─────────────────────────────────────────
# Muc nhieu = ty le flip nhan o VUNG BIEN; vung ro rang flip = muc/10.
# (nhieu phu thuoc dac trung — NNAR — sat thuc te hon flip deu, xem
#  Frenay & Verleysen 2014, Section III)
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30]

# ── Thi nghiem [3] COST-SENSITIVE ───────────────────────────────────────────
# Chi phi C(true, pred) cho operation_risk — don vi quy uoc.
# Bo sot High (du bao Low) dat gap 10 lan mot bao dong nham nhe.
COST_DEF = {
    ("High",   "Medium"): 5.0,   # bo sot mot phan — nguy hiem
    ("High",   "Low"):    10.0,  # bo sot hoan toan — nguy hiem nhat
    ("Medium", "Low"):    2.0,
    ("Medium", "High"):   1.0,   # bao dong nham — chi phi thap
    ("Low",    "Medium"): 1.0,
    ("Low",    "High"):   2.0,
}


# ── LOGGING ──────────────────────────────────────────────────────────────────

LINE  = "-" * 62
DLINE = "=" * 62


def _bar(pct, width=26):
    filled = int(width * max(0.0, min(1.0, float(pct))))
    return "[" + "#" * filled + "." * (width - filled) + f"] {float(pct)*100:5.1f}%"


def log(msg=""):
    # Console Windows co the la cp1252 — khong de UnicodeEncodeError
    # lam sap ca phien train.
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, errors="replace").decode(enc), flush=True)


def log_header(title):
    log(f"\n{DLINE}\n  {title}\n{DLINE}")


def log_section(title):
    log(f"\n{LINE}\n  {title}\n{LINE}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE-DEPENDENT LABEL NOISE (NNAR) — chi dung cho thi nghiem [2]
# Tiem vao TRAIN ONLY, test luon giu sach.
# ══════════════════════════════════════════════════════════════════════════════

def _composite_risk_score(df: pd.DataFrame) -> np.ndarray:
    """
    Diem rui ro tong hop (0→1) tinh tu nhieu features.
    Dung de xac dinh record nao nam o vung bien giua cac lop —
    noi nhan thuc te de bi gan sai nhat.
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


def flip_labels(y: np.ndarray, is_boundary: np.ndarray,
                boundary_rate: float, clear_rate: float,
                rng: np.random.Generator):
    """
    Flip nhan (da encode) theo ty le phu thuoc vung:
      boundary → boundary_rate, vung ro → clear_rate.
    Tra ve (y_noisy, so_nhan_bi_flip).
    """
    classes = np.unique(y)
    y_new   = y.copy()
    rates   = np.where(is_boundary, boundary_rate, clear_rate)
    flips   = rng.random(len(y)) < rates
    for i in np.where(flips)[0]:
        others   = classes[classes != y_new[i]]
        y_new[i] = rng.choice(others)
    return y_new, int(flips.sum())


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
    if "drone_id" not in df.columns:
        raise ValueError("Missing column 'drone_id' (can cho group split)")
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

    log("\n  Nhan giu SACH — khong tiem nhieu vao du lieu goc (v3).")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PER-DRONE EVALUATION — cach model van hanh tren tung drone
# ══════════════════════════════════════════════════════════════════════════════

def _per_drone_metrics(drone_ids_test, y_test, y_pred):
    """
    Danh gia model tren TUNG drone_id rieng (tren test set).
    Cho thay model hoat dong deu/khong deu tren toan ham doi.

    Tra ve dict: { drone_id: {drone_name, n_test, accuracy, f1} }
    """
    result = {}
    for drone in sorted(set(drone_ids_test.tolist())):
        mask = drone_ids_test == drone
        n    = int(mask.sum())
        if n == 0:
            continue
        acc  = accuracy_score(y_test[mask], y_pred[mask])
        f1   = f1_score(y_test[mask], y_pred[mask],
                        average="weighted", zero_division=0)
        result[str(drone)] = {
            "drone_name": DRONE_NAME_MAP.get(str(drone), "N/A"),
            "n_test":     n,
            "accuracy":   round(float(acc), 4),
            "f1":         round(float(f1),  4),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# THI NGHIEM [1] — GROUP CV THEO DRONE (unseen-drone generalization)
# ══════════════════════════════════════════════════════════════════════════════

def group_cv_eval(X_arr, y_arr, groups):
    """
    GroupKFold theo drone_id: moi fold test tren 2 drone mo hinh
    CHUA TUNG THAY khi train. Tra ve (fold_scores, fold_test_drones).
    """
    gkf         = GroupKFold(n_splits=CV_FOLDS)
    fold_scores = []
    fold_drones = []
    t0          = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(
            gkf.split(X_arr, y_arr, groups=groups), 1):
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_arr[tr_idx], y_arr[tr_idx])
        acc    = accuracy_score(y_arr[va_idx], clf.predict(X_arr[va_idx]))
        drones = sorted(set(groups[va_idx].tolist()))
        fold_scores.append(float(acc))
        fold_drones.append(drones)
        log(f"    Fold {fold_idx} (test={'+'.join(drones)}) : "
            f"{_bar(acc)}  {acc:.4f}")
    log(f"  Group CV done in {time.time()-t0:.1f}s")
    return fold_scores, fold_drones


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE TARGET (RF chinh + DT/LR so sanh + random CV + group CV)
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

    drone_ids = df["drone_id"].astype(str).values
    pos_idx   = np.arange(len(X))

    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, pos_idx, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    drone_ids_test = drone_ids[idx_test]
    log(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    X_arr = np.array(X_train)
    y_arr = np.array(y_train)

    # ── [CV-A] Random Stratified 5-Fold (baseline) ────────────────────────
    log(f"\n  [CV-A] Random Stratified {CV_FOLDS}-fold CV ...")
    skf            = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_fold_scores = []
    t0             = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        fold_clf = RandomForestClassifier(**RF_PARAMS)
        fold_clf.fit(X_arr[tr_idx], y_arr[tr_idx])
        fold_acc = accuracy_score(y_arr[va_idx], fold_clf.predict(X_arr[va_idx]))
        cv_fold_scores.append(float(fold_acc))
        log(f"    Fold {fold_idx} : {_bar(fold_acc)}  {fold_acc:.4f}")
    cv_scores = np.array(cv_fold_scores)
    log(f"  CV done in {time.time()-t0:.1f}s")
    log(f"  CV mean : {cv_scores.mean():.4f}  std : {cv_scores.std():.4f}")

    # ── [CV-B] GroupKFold theo drone_id (unseen-drone) ────────────────────
    log(f"\n  [CV-B] GroupKFold {CV_FOLDS}-fold theo drone_id "
        f"(test tren drone CHUA THAY) ...")
    X_full = np.array(X)
    g_scores, g_drones = group_cv_eval(X_full, y, drone_ids)
    g_arr = np.array(g_scores)
    gap   = float(cv_scores.mean() - g_arr.mean())
    log(f"  Group CV mean : {g_arr.mean():.4f}  std : {g_arr.std():.4f}")
    log(f"  Leakage gap (random - group) : {gap:+.4f}")
    if abs(gap) < 0.01:
        log("  -> Gap nho: model tong quat tot sang drone moi "
            "(nhan sinh tu luat chung, khong phu thuoc drone).")
    else:
        log("  -> Gap dang ke: random split da uoc luong QUA LAC QUAN.")

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

    # ── PER-DRONE: model van hanh tren tung drone ─────────────────────────
    per_drone = _per_drone_metrics(drone_ids_test, np.array(y_test), y_pred)
    log("\n  [RF] Accuracy tung drone (test set):")
    log(f"  {'Drone ID':<12} {'Model (DJI)':<18} "
        f"{'N test':>8} {'Accuracy':>9} {'F1':>8}")
    log(f"  {'-'*12} {'-'*18} {'-'*8} {'-'*9} {'-'*8}")
    for did, d in per_drone.items():
        log(f"  {did:<12} {d['drone_name']:<18} "
            f"{d['n_test']:>8,} {d['accuracy']:>9.4f} {d['f1']:>8.4f}")

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
        f"rec={rec_dt:.4f}  f1={f1_dt:.4f}")

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
        f"rec={rec_lr:.4f}  f1={f1_lr:.4f}")

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
        "evaluation_protocol": "clean_labels_v3",
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
        # CV-A: random stratified (baseline, co the lac quan do leakage)
        "cv_mean":             round(float(cv_scores.mean()), 4),
        "cv_std":              round(float(cv_scores.std()),  4),
        "cv_fold_scores":      [round(s, 4) for s in cv_fold_scores],
        # CV-B: GroupKFold theo drone_id (unseen-drone generalization)
        "group_cv": {
            "method":           f"GroupKFold({CV_FOLDS}) theo drone_id — "
                                "moi fold test tren 2 drone chua thay khi train",
            "mean":             round(float(g_arr.mean()), 4),
            "std":              round(float(g_arr.std()),  4),
            "fold_scores":      [round(s, 4) for s in g_scores],
            "fold_test_drones": g_drones,
            "gap_vs_random":    round(gap, 4),
        },
        # Per-class
        "per_class_f1":        {c: round(report_dict[c]["f1-score"],  4)
                                for c in classes if c in report_dict},
        "per_class_precision": {c: round(report_dict[c]["precision"], 4)
                                for c in classes if c in report_dict},
        "per_class_recall":    {c: round(report_dict[c]["recall"],    4)
                                for c in classes if c in report_dict},
        "confusion_matrix":    cm.tolist(),
        "train_time_s":        round(train_time, 2),
        # Per-drone: model van hanh tren tung drone (test set)
        "per_drone":           per_drone,
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

    # Tra ve them artefacts de tai su dung cho thi nghiem cost-sensitive
    artefacts = dict(model=model, le=le, X_test=np.array(X_test),
                     y_test=np.array(y_test))
    return metrics, artefacts


# ══════════════════════════════════════════════════════════════════════════════
# THI NGHIEM [2] — NOISE ROBUSTNESS STUDY
# Nhieu NNAR tiem vao TRAIN ONLY, test giu sach → do ben nhieu RF/DT/LR.
# ══════════════════════════════════════════════════════════════════════════════

def noise_robustness_study(df):
    log_header(f"THI NGHIEM [2] — NOISE ROBUSTNESS ({PRIMARY_TARGET})")
    log("  Thiet ke: nhieu nhan phu thuoc dac trung (NNAR) tiem vao")
    log("  TRAIN ONLY o cac muc 0/5/10/20/30%; TEST GIU SACH.")
    log("  Tham chieu: Frenay & Verleysen (2014), IEEE TNNLS.")

    X  = df[FEATURES]
    le = LabelEncoder()
    y  = le.fit_transform(df[PRIMARY_TARGET])
    high_idx = list(le.classes_).index("High") if "High" in le.classes_ else None

    pos_idx = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, _ = train_test_split(
        X, y, pos_idx, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    X_tr_arr = np.array(X_train)
    X_te_arr = np.array(X_test)

    # Vung bien tinh tren TRAIN (phu thuoc dac trung)
    scores_train = _composite_risk_score(df.iloc[idx_train])
    bnd_train    = _is_boundary(scores_train)
    log(f"\n  Train : {len(X_train):,}  |  Test (sach) : {len(X_test):,}")
    log(f"  Records vung bien (train) : {bnd_train.sum():,} "
        f"({bnd_train.mean()*100:.1f}%)")

    algos = [
        ("Random Forest",       lambda: RandomForestClassifier(**RF_PARAMS)),
        ("Decision Tree",       lambda: DecisionTreeClassifier(**DT_PARAMS)),
        ("Logistic Regression", lambda: LogisticRegression(**LR_PARAMS)),
    ]

    levels_out = []
    for level in NOISE_LEVELS:
        clear_rate = level / 10.0
        rng        = np.random.default_rng(NOISE_SEED)
        y_noisy, n_flipped = flip_labels(
            np.array(y_train), bnd_train, level, clear_rate, rng
        )
        eff = n_flipped / len(y_train)
        log(f"\n  -- Muc nhieu {level:.0%} (bien) / {clear_rate:.1%} (ro) "
            f"-> flip thuc te {eff:.1%} ({n_flipped:,} nhan) --")

        results = {}
        for algo_name, make in algos:
            t0  = time.time()
            clf = make()
            clf.fit(X_tr_arr, y_noisy)
            y_pred = clf.predict(X_te_arr)
            acc    = accuracy_score(y_test, y_pred)
            f1w    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            rec_hi = None
            if high_idx is not None:
                rec_all = recall_score(y_test, y_pred, average=None,
                                       zero_division=0)
                rec_hi  = float(rec_all[high_idx])
            results[algo_name] = {
                "accuracy":    round(float(acc), 4),
                "f1_weighted": round(float(f1w), 4),
                "recall_high": round(rec_hi, 4) if rec_hi is not None else None,
            }
            log(f"    {algo_name:<22} acc={acc:.4f}  f1={f1w:.4f}"
                + (f"  recall(High)={rec_hi:.4f}" if rec_hi is not None else "")
                + f"  ({time.time()-t0:.1f}s)")

        levels_out.append({
            "boundary_flip_rate":        level,
            "clear_flip_rate":           round(clear_rate, 4),
            "effective_train_noise_pct": round(eff * 100, 2),
            "n_flipped":                 n_flipped,
            "results":                   results,
        })

    out = {
        "target":        PRIMARY_TARGET,
        "design":        "Nhieu NNAR (phu thuoc dac trung, vung bien flip nhieu hon) "
                         "tiem vao TRAIN ONLY; test set giu nhan sach.",
        "reference":     "Frenay & Verleysen (2014). Classification in the presence "
                         "of label noise: a survey. IEEE TNNLS 25(5).",
        "noise_seed":    NOISE_SEED,
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "classes":       list(le.classes_),
        "levels":        levels_out,
        "trained_at":    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(MODEL_DIR / "noise_robustness.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("\n  Saved : noise_robustness.json")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# THI NGHIEM [3] — COST-SENSITIVE DECISION (Elkan 2001)
# Dung predict_proba cua RF final + ma tran chi phi → quy tac Bayes
# toi thieu chi phi ky vong. Khong can fit them model nao.
# ══════════════════════════════════════════════════════════════════════════════

def _eval_decisions(y_true, y_pred, C, high_idx):
    acc     = accuracy_score(y_true, y_pred)
    rec_all = recall_score(y_true, y_pred, average=None, zero_division=0)
    cost    = float(C[y_true, y_pred].mean())
    return {
        "accuracy":    round(float(acc), 4),
        "recall_high": round(float(rec_all[high_idx]), 4),
        "mean_cost":   round(cost, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def cost_sensitive_study(artefacts):
    log_header(f"THI NGHIEM [3] — COST-SENSITIVE DECISION ({PRIMARY_TARGET})")
    log("  Quy tac Bayes: du bao lop j* = argmin_j sum_i P(i|x)*C(i,j)")
    log("  Tham chieu: Elkan (2001), The foundations of cost-sensitive")
    log("  learning, IJCAI.")

    model  = artefacts["model"]
    le     = artefacts["le"]
    X_test = artefacts["X_test"]
    y_test = artefacts["y_test"]

    classes  = list(le.classes_)            # thu tu cua LabelEncoder
    high_idx = classes.index("High")
    n        = len(classes)

    # Ma tran chi phi theo dung thu tu classes cua LabelEncoder
    C = np.zeros((n, n))
    for i, ti in enumerate(classes):
        for j, pj in enumerate(classes):
            C[i, j] = COST_DEF.get((ti, pj), 0.0)

    log(f"\n  Classes (LE order): {classes}")
    log("  Cost matrix C(true -> pred):")
    cw = max(len(c) for c in classes) + 2
    log("  " + " " * cw + "".join(f"{c:>{cw}}" for c in classes))
    for i, row in enumerate(C):
        log("  " + f"{classes[i]:<{cw}}" + "".join(f"{v:>{cw}.1f}" for v in row))

    proba = model.predict_proba(X_test)

    # (a) Baseline: argmax (quy tac hien tai cua he thong)
    y_argmax = proba.argmax(axis=1)
    base     = _eval_decisions(y_test, y_argmax, C, high_idx)

    # (b) Bayes cost-minimizing rule (Elkan 2001) — khong co sieu tham so
    exp_cost = proba @ C          # (n_samples, n_classes): chi phi ky vong
    y_bayes  = exp_cost.argmin(axis=1)
    bayes    = _eval_decisions(y_test, y_bayes, C, high_idx)

    log(f"\n  {'Quy tac':<28} {'Accuracy':>9} {'Recall(High)':>13} {'Mean cost':>10}")
    log(f"  {'-'*28} {'-'*9} {'-'*13} {'-'*10}")
    log(f"  {'Argmax (baseline)':<28} {base['accuracy']:>9.4f} "
        f"{base['recall_high']:>13.4f} {base['mean_cost']:>10.4f}")
    log(f"  {'Bayes cost-minimizing':<28} {bayes['accuracy']:>9.4f} "
        f"{bayes['recall_high']:>13.4f} {bayes['mean_cost']:>10.4f}")
    log(f"\n  -> Recall(High) thay doi : {base['recall_high']:.4f} -> "
        f"{bayes['recall_high']:.4f} "
        f"({(bayes['recall_high']-base['recall_high'])*100:+.2f} diem %)")
    log(f"  -> Chi phi ky vong/chuyen bay : {base['mean_cost']:.4f} -> "
        f"{bayes['mean_cost']:.4f} "
        f"({(bayes['mean_cost']/base['mean_cost']-1)*100:+.1f}%)")

    # (c) Threshold sweep (mo ta do nhay): du bao High neu P(High) >= tau
    sweep = []
    other_cols = [j for j in range(n) if j != high_idx]
    proba_other = proba[:, other_cols]
    for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
        y_thr = np.array(other_cols)[proba_other.argmax(axis=1)]
        y_thr = np.where(proba[:, high_idx] >= tau, high_idx, y_thr)
        ev    = _eval_decisions(y_test, y_thr, C, high_idx)
        sweep.append({
            "tau":         float(tau),
            "accuracy":    ev["accuracy"],
            "recall_high": ev["recall_high"],
            "mean_cost":   ev["mean_cost"],
        })

    out = {
        "target":     PRIMARY_TARGET,
        "reference":  "Elkan (2001). The foundations of cost-sensitive learning. "
                      "IJCAI. Quy tac Bayes: argmin_j Σ_i P(i|x)·C(i,j).",
        "classes":    classes,
        "cost_matrix": C.tolist(),
        "cost_definition_note": "C(true,pred): bo sot High (pred Low) = 10, "
                                "(pred Medium) = 5; bao dong nham = 1-2.",
        "baseline_argmax":      base,
        "bayes_cost_minimizing": bayes,
        "threshold_sweep":      sweep,
        "test_samples":         int(len(y_test)),
        "trained_at":           pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(MODEL_DIR / "cost_sensitive.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("\n  Saved : cost_sensitive.json")
    log("  -> core.predict_drone se tu dong ap dung quy tac Bayes nay "
        "khi file ton tai.")
    return out


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
    log_header("DRONE DSS — MODEL TRAINING v3 | DSS301")
    log(f"  n_estimators       : {RF_PARAMS['n_estimators']}")
    log(f"  CV folds           : {CV_FOLDS} (random) + {CV_FOLDS} (group theo drone)")
    log(f"  Test size          : {TEST_SIZE:.0%}")
    log(f"  class_weight       : {RF_PARAMS['class_weight']}")
    log(f"  n_jobs             : NOT USED (Python 3.14 safe)")
    log(f"  evaluation         : clean_labels_v3 (khong tiem nhieu vao test)")
    log(f"  noise study        : {[f'{l:.0%}' for l in NOISE_LEVELS]} tren "
        f"{PRIMARY_TARGET} (train-only)")
    log(f"  cost-sensitive     : Elkan (2001) Bayes rule tren {PRIMARY_TARGET}")
    log(f"  comparison models  : Decision Tree + Logistic Regression")

    wall_start = time.time()

    df = load_data()
    X  = df[FEATURES]

    log_header("Training 3 Targets (RF + DT/LR, random CV + group CV)")
    all_importances = []
    summary         = []
    risk_artefacts  = None

    for target_col, model_name in TARGETS:
        m, art = train_one(df, target_col, model_name, X, all_importances)
        summary.append(m)
        if target_col == PRIMARY_TARGET:
            risk_artefacts = art

    log_section("Feature Importance Export")
    save_feature_importance(all_importances)

    # ── 2 thi nghiem nghien cuu ──────────────────────────────────────────
    noise_robustness_study(df)
    cost_sensitive_study(risk_artefacts)

    total_time = time.time() - wall_start
    log_header("Training Complete")
    log(f"\n  {'Target':<26} {'RF Acc':>8} {'RandCV':>8} "
        f"{'GroupCV':>8} {'Gap':>7} {'DT Acc':>8} {'LR Acc':>8}")
    log(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    for m in summary:
        dt_acc = m["comparison"]["Decision Tree"]["accuracy"]
        lr_acc = m["comparison"]["Logistic Regression"]["accuracy"]
        log(f"  {m['target']:<26} {m['accuracy']:>8.4f} "
            f"{m['cv_mean']:>8.4f} {m['group_cv']['mean']:>8.4f} "
            f"{m['group_cv']['gap_vs_random']:>+7.4f} "
            f"{dt_acc:>8.4f} {lr_acc:>8.4f}")

    log(f"\n  Total time : {total_time:.1f}s")
    log(f"\n  Files in Model/:")
    for p in sorted(MODEL_DIR.iterdir()):
        log(f"    {p.name:<50} {p.stat().st_size/1024:>7.1f} KB")
    log(f"\n  App: streamlit run app.py\n")
    log(f"{DLINE}\n")


if __name__ == "__main__":
    main()
