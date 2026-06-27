"""
app_views/batch_predict.py — Batch Prediction Module
======================================================
Xử lý dữ liệu hàng loạt từ file CSV, tự động làm sạch và dự đoán.
"""

import streamlit as st
import pandas as pd
import joblib

from ui import render_top_nav, render_page_title, render_section_label

# ── Nạp Model (Dùng Cache để tránh load lại nhiều lần gây giật lag) ──
@st.cache_resource
def load_ml_pipeline():
    try:
        # Load Models
        risk_model = joblib.load("Model/operation_risk_model.joblib")
        maint_model = joblib.load("Model/maintenance_action_model.joblib")
        recom_model = joblib.load("Model/recommendation_model.joblib")

        # Load Label Encoders
        risk_le = joblib.load("Model/operation_risk_model_label_encoder.joblib")
        maint_le = joblib.load("Model/maintenance_action_model_label_encoder.joblib")
        recom_le = joblib.load("Model/recommendation_model_label_encoder.joblib")

        return (risk_model, risk_le), (maint_model, maint_le), (recom_model, recom_le)
    except Exception as e:
        return None, str(e)


def validate_and_format_data(df_raw):
    """Làm sạch và ép định dạng 10 features chuẩn từ file tải lên"""
    EXPECTED_FEATURES = [
        "battery_level", "flight_time", "signal_strength", "temperature",
        "wind_speed", "gps_accuracy", "altitude", "speed", "humidity", "pressure"
    ]

    df = df_raw.copy()

    # 1. Chuẩn hóa tên cột (chữ thường, thay khoảng trắng bằng gạch dưới)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # 2. Kiểm tra cột bắt buộc (Fail-Fast)
    missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_cols:
        return None, f"File thiếu các cột dữ liệu bắt buộc: {', '.join(missing_cols)}"

    # 3. Ép đúng thứ tự cột mà model cần
    df_clean = df[EXPECTED_FEATURES]

    # 4. Ép kiểu Float (Bắt lỗi nếu có ký tự lạ không thể ép thành số)
    try:
        df_clean = df_clean.astype(float)
    except ValueError:
        return None, "Dữ liệu chứa ký tự không hợp lệ (ví dụ: chữ cái, ký hiệu '%', 'm/s'). Vui lòng làm sạch các ô chỉ còn con số trước khi tải lên."

    return df_clean, "Success"


def render():
    render_top_nav()
    render_page_title(
        "Dự đoán hàng loạt (Batch Prediction)",
        "Tải lên file CSV chứa dữ liệu cảm biến thô để tự động phân tích và đánh giá rủi ro cho toàn bộ hạm đội."
    )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # ── VÙNG TẢI FILE ──
    with st.container(border=True):
        uploaded_file = st.file_uploader("Kéo thả file .CSV dữ liệu Drone vào đây", type=["csv"])
        st.caption("💡 *Lưu ý: File CSV cần chứa đủ 10 cột thông số cảm biến (battery_level, wind_speed, temperature...). Tên cột có thể lộn xộn vị trí, hệ thống sẽ tự động sắp xếp lại.*")

    if uploaded_file is not None:
        st.divider()
        render_section_label("Tiến trình Xử lý & Dự đoán")

        with st.status("Đang phân tích dữ liệu...", expanded=True) as status:
            try:
                # 1. Đọc file
                st.write("Đang đọc file CSV...")
                df_raw = pd.read_csv(uploaded_file)

                # 2. Xử lý định dạng
                st.write("Đang làm sạch và đối chiếu cấu trúc dữ liệu...")
                df_clean, msg = validate_and_format_data(df_raw)

                if df_clean is None:
                    status.update(label="Thất bại", state="error", expanded=True)
                    st.error(msg)
                    return

                # 3. Load Model
                st.write("Đang khởi động Random Forest Pipeline...")
                pipeline_data = load_ml_pipeline()
                if pipeline_data[0] is None:
                    status.update(label="Lỗi Model", state="error", expanded=True)
                    st.error(f"Không thể tải Model từ thư mục 'Model/'. Chi tiết: {pipeline_data[1]}")
                    return

                (risk_model, risk_le), (maint_model, maint_le), (recom_model, recom_le) = pipeline_data

                # 4. Dự đoán
                st.write("Đang chạy mô hình dự đoán...")
                risk_preds = risk_model.predict(df_clean)
                maint_preds = maint_model.predict(df_clean)
                recom_preds = recom_model.predict(df_clean)

                # 5. Decode kết quả (Chuyển từ số về nhãn chữ)
                st.write("Đang giải mã nhãn dự đoán...")
                df_raw['PREDICTED_Operation_Risk'] = risk_le.inverse_transform(risk_preds)
                df_raw['PREDICTED_Maintenance_Action'] = maint_le.inverse_transform(maint_preds)
                df_raw['PREDICTED_Recommendation'] = recom_le.inverse_transform(recom_preds)

                status.update(label=f"Hoàn tất xử lý {len(df_raw):,} bản ghi!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Lỗi không xác định", state="error", expanded=True)
                st.error(f"Đã xảy ra lỗi hệ thống: {str(e)}")
                return

        # ── HIỂN THỊ KẾT QUẢ & NÚT DOWNLOAD ──
        st.success("✅ Toàn bộ dữ liệu đã được phân tích thành công!")

        st.markdown("#### 🔍 Bản xem trước kết quả")
        # Hiển thị 10 dòng đầu tiên để user kiểm tra
        st.dataframe(df_raw.head(10), use_container_width=True)

        # Nút tải file CSV
        csv_data = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Tải File Kết Quả Phân Tích (.CSV)",
            data=csv_data,
            file_name="drone_batch_predictions.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )