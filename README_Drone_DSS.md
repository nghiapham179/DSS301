# 🚁 Drone DSS — Hệ Thống Hỗ Trợ Ra Quyết Định Bảo Hành & Vận Hành Drone

> **Môn học:** DSS301 — Decision Support Systems
> **Trường:** FPT University
> **Công nghệ:** Python · scikit-learn · Streamlit · Plotly

---

## Mô tả dự án

**Drone DSS** là Hệ thống Hỗ trợ Ra quyết định (DSS) kiến trúc **hybrid** cho vận hành và bảo trì
đội drone, kết hợp 3 tầng ra quyết định:

| Tầng | Vai trò | Cơ sở |
|---|---|---|
| **1. Data-driven (ML)** | 3 mô hình Random Forest dự đoán `operation_risk` / `maintenance_action` / `recommendation`; riêng risk dùng **quy tắc Bayes tối thiểu chi phí kỳ vọng** thay argmax — ưu tiên không bỏ sót High-risk | Elkan (2001), *IJCAI* |
| **2. Knowledge-driven (Rule)** | Ngưỡng an toàn cứng **theo từng dòng drone** (DJI Mini 3 / Air 3 / Mavic 3 Pro) từ spec chính hãng; luôn **ghi đè** ML khi vi phạm | Power (2002); spec DJI |
| **3. Trend (giám sát tuần tự)** | Trang Phiên bay trực tiếp: EWMA control chart trên risk score + dự báo pin tick kế → quyết định **chủ động** trước khi vi phạm | Roberts (1959), *Technometrics* |

### 5 trang chức năng

1. **🏠 Dashboard** — tổng quan hạm đội: KPI, phân phối rủi ro, bảo trì, top drone rủi ro.
2. **🎯 Dự đoán** — 3 tab: mô phỏng Slider (kèm chọn hồ sơ dòng drone + radar chart), Form nhập
   thực địa (tự nhận dòng máy, lưu 49 cột chuẩn), Batch CSV (dự đoán hàng loạt).
3. **🛫 Phiên bay** — phiên bay trực tiếp: mỗi chu kỳ 3 phút (demo: 10 giây) nhận **1 dòng telemetry
   đúng format 49 cột** và phân tích tức thời: bay tiếp / giám sát / quay về / hạ cánh ngay.
4. **📊 Phân tích drone** — drill-down lịch sử từng drone.
5. **🤖 Model Info** — 4 tab: so sánh RF/DT/LR, chi tiết metrics + confusion matrix, feature
   importance, và **🧪 Nghiên cứu** (3 thí nghiệm đánh giá nâng cao).

---

## Dữ liệu

| Thông số | Giá trị |
|---|---|
| Tổng bản ghi | 200,000 (synthetic) + dữ liệu người dùng nhập |
| Số drone | 10 (ánh xạ 3 dòng DJI) |
| Số cột chuẩn | 49 |
| Cột đầu vào (features) | 10 cột cảm biến |
| Cột đầu ra (targets) | 3 model ML + 1 rule engine |

**10 features đầu vào:**
`battery_level` · `flight_time` · `signal_strength` · `temperature` · `wind_speed` · `gps_accuracy` · `altitude` · `speed` · `humidity` · `pressure`

**Phân phối nhãn `operation_risk`:** Medium 53.7% · Low 27.6% · High 18.8%

---

## Mô hình ML & Thiết kế đánh giá (v3 — Honest Evaluation)

Random Forest (`n_estimators=100`, `class_weight="balanced"`, stratified split 80/20),
so sánh cùng Decision Tree và Logistic Regression.

> ⚠️ **Lưu ý học thuật:** nhãn của dataset là synthetic, sinh **tất định từ chính các features**
> bằng bộ luật nghiệp vụ → accuracy ~99.9% trên random split là hệ quả tất yếu (mô hình "học lại
> bộ luật" — *label leakage*, Kapoor & Narayanan 2023, *Patterns*), không phản ánh năng lực trên
> dữ liệu thật. Vì vậy hệ thống **không** hạ accuracy bằng cách tiêm nhiễu vào test (cách làm sai
> về đo lường), mà giữ nhãn sạch và trả lời 3 câu hỏi nghiên cứu trung thực:

### Kết quả huấn luyện (test 40,000 bản ghi)

| Target | RF Acc | Random CV | **GroupKFold theo drone** | Leakage gap | DT | LR |
|---|---|---|---|---|---|---|
| operation_risk | 99.94% | 99.92% | 99.94% | −0.02đ% | 99.91% | 76.52% |
| maintenance_action | 99.91% | 99.88% | 99.92% | −0.04đ% | 99.79% | 67.81% |
| recommendation | 99.93% | 99.92% | 99.93% | −0.01đ% | 99.87% | 77.00% |

### 3 thí nghiệm nghiên cứu (xem app: Model Info → 🧪 Nghiên cứu)

| # | Thí nghiệm | Phương pháp | Kết quả chính | Cơ sở nghiên cứu |
|---|---|---|---|---|
| 1 | **Tổng quát hóa sang drone mới** | GroupKFold 5-fold theo `drone_id` (mỗi fold test trên 2 drone chưa thấy) so với Random CV | Leakage gap ≈ 0 → không có overfitting theo thiết bị; đồng thời là bằng chứng định lượng cho hạn chế của nhãn synthetic (sinh từ luật chung, không đặc thù drone) | Kapoor & Narayanan (2023); Roberts et al. (2021, *Nat. Mach. Intell.*) |
| 2 | **Độ bền nhiễu nhãn** | Nhiễu phụ thuộc đặc trưng (NNAR — vùng biên flip nhiều hơn) tiêm vào **train only** ở mức 0/5/10/20/30%, test giữ **sạch** | Ở 30% nhiễu biên (17.3% nhãn flip): RF −0.29đ% vs DT −0.93đ% → RF bền nhiễu nhất; LR "phẳng" chỉ vì baseline thấp (~76%) | Frénay & Verleysen (2014, *IEEE TNNLS*) |
| 3 | **Quyết định nhạy chi phí** | Ma trận chi phí bất đối xứng (bỏ sót High = 10× báo động nhầm) + quy tắc Bayes *argmin<sub>j</sub> Σ<sub>i</sub> P(i\|x)·C(i,j)* thay argmax — **áp dụng thật trong trang Dự đoán & Phiên bay** | Recall(High): 99.96% → **100%**; đổi bằng accuracy −0.61đ% (báo động nhầm tăng) — trên nhãn sạch mô hình đã gần hoàn hảo nên lợi ích cost-sensitive chỉ rõ khi có bất định; quy tắc Elkan giả định xác suất đã calibrate → calibration là hướng phát triển tiếp | Elkan (2001, *IJCAI*); Niculescu-Mizil & Caruana (2005, *ICML*) |

**Output:** `Model/*_metrics.json` (kèm `group_cv`) · `Model/noise_robustness.json` · `Model/cost_sensitive.json`

---

## Hồ sơ quy tắc vận hành theo dòng drone (tầng Knowledge-driven)

Tầng quy tắc (`flight_decision` trong `core.py`) áp **ngưỡng riêng theo từng dòng drone**,
suy từ thông số kỹ thuật chính hãng DJI:

| Ngưỡng | DJI Mini 3 (248g) | DJI Air 3 (720g) | Mavic 3 Pro (958g) | Cơ sở |
|---|---|---|---|---|
| Gió tối đa (cấm bay) | 10.7 m/s | 12 m/s | 12 m/s | Max wind resistance — spec DJI |
| Dải nhiệt vận hành | −10…40°C | −10…40°C | −10…40°C | Operating temperature — spec DJI |
| Pin quay về trạm (RTH) | 30% | 25% | 20% | Nguyên tắc dự phòng năng lượng: drone càng nhẹ càng bị gió tiêu hao pin nhanh |
| Pin khẩn cấp (cấm bay) | 15% | 12% | 10% | Nới từ mức cảnh báo low-battery của DJI Fly theo khối lượng |
| Thời lượng bay tối đa | 38 phút | 46 phút | 43 phút | Max flight time — spec DJI |
| Tốc độ tối đa | 16 m/s | 21 m/s | 21 m/s | Max speed (S-mode) — spec DJI |
| Trần bay | 120 m | 120 m | 120 m | Điều kiện cấp phép bay UAV dân dụng phổ biến tại Việt Nam |

Khi ML nói Low/Medium nhưng thông số vi phạm ngưỡng cứng của dòng máy, hệ thống ưu tiên tầng quy
tắc và **hiển thị rõ việc ghi đè** — kiến trúc hybrid DSS (knowledge-driven > data-driven cho ràng
buộc an toàn; Power 2002). Hồ sơ "Mặc định (ngưỡng chung)" giữ bộ ngưỡng cũ, tương thích thang đo
dataset synthetic (vd. gió 0–50 m/s).

---

## Phiên bay trực tiếp (tầng Trend — giám sát tuần tự)

Trang **🛫 Phiên bay** mô phỏng luồng telemetry vận hành thật đổ về DSS:

- Bấm **Bắt đầu phiên bay** → mỗi chu kỳ (**3 phút/tick**, chế độ demo **10 giây/tick**) hệ thống
  nhận 1 dòng dữ liệu **đúng format 49 cột** (simulator suy biến trạng thái: pin hao theo gió/tốc
  độ, gió AR(1), vị trí GPS di chuyển theo heading) và phân tích ngay bằng pipeline sẵn có.
- **Verdict mỗi tick:** ✅ Bay tiếp / 👁️ Giám sát / 🔙 Quay về trạm / 🛑 Hạ cánh ngay, với:
  - **Leo thang:** 2 tick cảnh báo liên tiếp → tự nâng thành Quay về;
  - **Quay về chủ động (proactive):** dự báo pin tick kế < ngưỡng RTH của dòng máy → khuyến nghị
    quay về **trước khi** vi phạm;
  - **EWMA control chart** (λ=0.3) trên risk score phát hiện drift rủi ro tăng dần.
- UI trực tiếp: biểu đồ pin/gió/EWMA theo tick, **bản đồ đường bay**, nhật ký telemetry, 3 nút
  **tiêm sự cố** (gió giật / suy hao tín hiệu / sụt pin) để kiểm thử phản ứng hệ thống.
- Mỗi tick được lưu qua `save_custom` → xuất hiện ngay trong Dashboard & Phân tích drone, kèm bản
  sao `Data/live_session_log.csv` và nút tải nhật ký phiên.

---

## Cấu trúc thư mục

```
DSS_Drone_301/
├── app.py                        # Entry point — page config + sidebar nav + routing
├── core.py                       # Logic dùng chung: loaders, predict (cost-sensitive),
│                                 #   flight_decision (hồ sơ theo dòng drone), save_custom
├── ui.py                         # CSS và UI components
├── train_model.py                # Train v3 + 2 thí nghiệm nghiên cứu (nhiễu, chi phí)
│
├── app_views/
│   ├── dashboard.py              # 🏠 System Overview
│   ├── parameters.py             # 🎯 Dự đoán (Slider / Form / Batch CSV)
│   ├── live_flight.py            # 🛫 Phiên bay trực tiếp (telemetry 3 phút/tick)
│   ├── analysis.py               # 📊 Phân tích theo drone
│   └── model_info.py             # 🤖 Metrics + tab 🧪 Nghiên cứu
│
├── Data/
│   ├── drone_data_clean.csv      # Dataset chính (200,000 records × 49 cột)
│   ├── custom_drone_data.csv     # Dữ liệu người dùng nhập + telemetry phiên bay (tự tạo)
│   └── live_session_log.csv      # Bản sao nhật ký các phiên bay trực tiếp (tự tạo)
│
├── Model/                        # Sinh bởi train_model.py
│   ├── operation_risk_model.joblib          (+ _label_encoder, _metrics.json)
│   ├── maintenance_action_model.joblib      (+ _label_encoder, _metrics.json)
│   ├── recommendation_model.joblib          (+ _label_encoder, _metrics.json)
│   ├── feature_importance.csv
│   ├── noise_robustness.json     # Thí nghiệm độ bền nhiễu (train-only noise)
│   └── cost_sensitive.json       # Ma trận chi phí + quy tắc Bayes (Elkan 2001)
│
├── Requirement.txt
└── README_Drone_DSS.md
```

---

## Hướng dẫn cài đặt & chạy

### Yêu cầu hệ thống

- Python **3.11 trở lên** (đã kiểm chứng trên 3.14 — train script tắt `n_jobs` để an toàn)
- RAM tối thiểu 4GB (dataset 200K records)

### Bước 1 — Cài thư viện

```bash
pip install streamlit pandas scikit-learn joblib plotly numpy
```

Kiểm tra: `python -c "import streamlit, sklearn, joblib, plotly; print('OK')"`

### Bước 2 — Đặt file data

Đảm bảo `Data/drone_data_clean.csv` tồn tại (xem cấu trúc thư mục ở trên).

### Bước 3 — Train model

```bash
python train_model.py
```

> ⏱️ Khoảng **30–45 phút**: 3 targets × (5-fold random CV + 5-fold GroupKFold + train final
> + DT/LR) + thí nghiệm nhiễu 5 mức + phân tích cost-sensitive. Khi xong, `Model/` chứa đầy đủ
> model + 5 file kết quả nghiên cứu.

### Bước 4 — Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt mở tại `http://localhost:8501`.

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `No module named 'streamlit'` | Thư viện cài vào Python khác với interpreter đang chạy | Kiểm tra **Settings → Python Interpreter** trỏ đúng `python.exe` đã cài thư viện |
| `FileNotFoundError: Data/drone_data_clean.csv` | Thiếu file dataset | Tạo thư mục `Data/` và copy file CSV vào |
| `FileNotFoundError: Model/...joblib` | Chưa train | Chạy `python train_model.py` trước |
| `UnicodeEncodeError` khi train trên Windows | Console cp1252 | Đã xử lý trong code; nếu vẫn gặp: `set PYTHONIOENCODING=utf-8` trước khi chạy |
| Tab 🧪 Nghiên cứu báo "Chưa có ..." | Model train bằng script cũ (v2) | Chạy lại `python train_model.py` (v3) để sinh `group_cv` + 2 file JSON nghiên cứu |
| Phiên bay không tự tick | Tab trình duyệt bị ẩn/đóng | Giữ tab mở — `st.fragment(run_every=...)` chỉ chạy khi tab hoạt động |

---

## Công nghệ sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `streamlit` | ≥ 1.37 (khuyến nghị 1.58) | Web UI + `st.fragment(run_every)` cho phiên bay trực tiếp |
| `scikit-learn` | ≥ 1.3 | Random Forest, GroupKFold, LabelEncoder, metrics |
| `pandas` | ≥ 2.0 | Xử lý dữ liệu |
| `joblib` | ≥ 1.3 | Lưu/load model |
| `plotly` | ≥ 5.0 | Biểu đồ interactive |
| `numpy` | ≥ 1.24 | Tính toán số học, simulator AR(1) |

---

## Tài liệu tham khảo

1. Elkan, C. (2001). *The foundations of cost-sensitive learning.* IJCAI.
2. Frénay, B., & Verleysen, M. (2014). *Classification in the presence of label noise: a survey.* IEEE Transactions on Neural Networks and Learning Systems, 25(5).
3. Kapoor, S., & Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns, 4(9).
4. Roberts, S. W. (1959). *Control chart tests based on geometric moving averages.* Technometrics, 1(3). (EWMA)
5. Roberts, M., et al. (2021). *Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19.* Nature Machine Intelligence, 3.
6. Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good probabilities with supervised learning.* ICML.
7. Power, D. J. (2002). *Decision Support Systems: Concepts and Resources for Managers.* Quorum Books.
8. DJI. *Mini 3 / Air 3 / Mavic 3 Pro — Specifications.* dji.com.

---

## Tác giả

**Nhóm DSS301 — FPT University**
GitHub: [https://github.com/nghiapham179/DSS301](https://github.com/nghiapham179/DSS301)

---

*README này được tạo cho mục đích học thuật — môn DSS301, FPT University.*
