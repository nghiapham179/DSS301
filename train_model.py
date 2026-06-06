import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path


# ================== PATH CONFIG ==================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "drone_data_clean.csv"
MODEL_DIR = BASE_DIR / "Model"

# Tạo thư mục Model nếu chưa có
MODEL_DIR.mkdir(exist_ok=True)


# ================== LOAD DATA ==================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DRONE DSS MODEL TRAINING")
print("=" * 60)
print(f"Đã đọc data: {df.shape[0]} dòng, {df.shape[1]} cột")
print(f"Data path: {DATA_PATH}")


# ================== FEATURES ==================

FEATURES = [
    "battery_level",
    "flight_time",
    "signal_strength",
    "temperature",
    "wind_speed",
    "gps_accuracy",
    "altitude",
    "speed",
    "humidity",
    "pressure"
]

TARGETS = [
    "operation_risk",
    "maintenance_action",
    "recommendation"
]


# ================== CHECK COLUMNS ==================

missing_features = [col for col in FEATURES if col not in df.columns]
missing_targets = [col for col in TARGETS if col not in df.columns]

if missing_features:
    raise ValueError(f"Thiếu cột input trong dataset: {missing_features}")

if missing_targets:
    raise ValueError(f"Thiếu cột target trong dataset: {missing_targets}")

X = df[FEATURES]


# ================== TRAIN FUNCTION ==================

def train_and_save(target_col, model_file_name):
    print("\n" + "-" * 60)
    print(f"Đang train model cho target: {target_col}")
    print("-" * 60)

    # Encode label dạng chữ thành số
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Đánh giá model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # Lưu model và label encoder
    model_path = MODEL_DIR / f"{model_file_name}.joblib"
    encoder_path = MODEL_DIR / f"{model_file_name}_label_encoder.joblib"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"Đã lưu model: {model_path}")
    print(f"Đã lưu encoder: {encoder_path}")


# ================== TRAIN 3 MODELS ==================

train_and_save(
    target_col="operation_risk",
    model_file_name="operation_risk_model"
)

train_and_save(
    target_col="maintenance_action",
    model_file_name="maintenance_action_model"
)

train_and_save(
    target_col="recommendation",
    model_file_name="recommendation_model"
)

print("\n" + "=" * 60)
print("HOÀN THÀNH TRAIN MODEL")
print("=" * 60)
print("Các file đã được lưu vào thư mục Model/")