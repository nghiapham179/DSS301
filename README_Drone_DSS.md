# 🚁 Drone DSS — Hệ Thống Hỗ Trợ Ra Quyết Định Bảo Hành & Vận Hành Drone

> **Môn học:** DSS301 — Decision Support Systems  
> **Trường:** FPT University  
> **Công nghệ:** Python · scikit-learn · Streamlit · Plotly  

---

## Mô tả dự án

**Drone DSS** là một hệ thống hỗ trợ ra quyết định (Decision Support System) ứng dụng Machine Learning để phân tích dữ liệu cảm biến UAV (drone) theo thời gian thực, từ đó đưa ra các khuyến nghị về:

- **Mức độ rủi ro vận hành** — Low / Medium / High
- **Hành động bảo trì** — Monitor / Inspection recommended / Maintenance required / No maintenance needed
- **Khuyến nghị cụ thể** — 5 cấp độ từ "tiếp tục bay" đến "không bay, cần bảo trì ngay"
- **Trạng thái bay** — Đủ điều kiện bay / Bay kèm giám sát / Quay về trạm / Yêu cầu bảo trì / Cấm bay

### Dữ liệu

| Thông số | Giá trị |
|---|---|
| Tổng bản ghi | 200,000 |
| Số drone | 10 |
| Số cột | 24 |
| Cột đầu vào (features) | 10 cột cảm biến |
| Cột đầu ra (targets) | 3 model ML + 1 rule engine |

**10 features đầu vào:**
`battery_level` · `flight_time` · `signal_strength` · `temperature` · `wind_speed` · `gps_accuracy` · `altitude` · `speed` · `humidity` · `pressure`

**Phân phối nhãn:**

| operation_risk | Số lượng | Tỷ lệ |
|---|---|---|
| Medium | 107,323 | 53.7% |
| Low | 55,175 | 27.6% |
| High | 37,502 | 18.8% |

### Mô hình ML

Hệ thống sử dụng **Random Forest Classifier** với các tham số:
- `n_estimators = 100`
- `class_weight = "balanced"` — xử lý mất cân bằng nhãn
- `stratify = y` — đảm bảo tập test có cùng tỷ lệ class
- Test size: 20% (40,000 bản ghi)

**Kết quả (accuracy trên tập test):**

| Model | Target | Accuracy |
|---|---|---|
| Model 1 | `operation_risk` | ~99.9% |
| Model 2 | `maintenance_action` | ~99.9% |
| Model 3 | `recommendation` | ~99.9% |

---

## Cấu trúc thư mục

```
DSS301/
├── app.py                        # File chạy chính — Streamlit app
├── train_model.py                # Script train và lưu model
├── ui.py                         # CSS và UI components
│
├── Data/
│   ├── drone_data_clean.csv      # Dataset chính (200,000 records)
│   └── custom_drone_data.csv     # Dữ liệu người dùng nhập (tự tạo)
│
├── Model/
│   ├── operation_risk_model.joblib
│   ├── operation_risk_model_label_encoder.joblib
│   ├── maintenance_action_model.joblib
│   ├── maintenance_action_model_label_encoder.joblib
│   ├── recommendation_model.joblib
│   └── recommendation_model_label_encoder.joblib
│
├── requirements.txt
└── README.md
```

---

## Hướng dẫn cài đặt & chạy

### Yêu cầu hệ thống

- Python **3.9 trở lên** (khuyến nghị 3.11)
- PyCharm hoặc VS Code
- RAM tối thiểu 4GB (do dataset 200K records)

> ⚠️ **Lưu ý Python 3.14:** Nếu dùng Python 3.14, joblib có thể gặp lỗi parallel processing. Khuyến nghị dùng Python 3.11.

---

### Bước 1 — Clone hoặc tải repo

**Cách 1 — Git clone:**
```bash
git clone https://github.com/nghiapham179/DSS301.git
cd DSS301
```

**Cách 2 — Download ZIP:**
1. Vào `https://github.com/nghiapham179/DSS301`
2. Click **Code → Download ZIP**
3. Giải nén vào thư mục bất kỳ

---

### Bước 2 — Mở project trong PyCharm

1. Mở PyCharm → **File → Open** → chọn thư mục `DSS301`
2. PyCharm sẽ tự nhận project

---

### Bước 3 — Cài thư viện

Mở **Terminal** trong PyCharm (tab dưới cùng) và chạy:

```bash
pip install streamlit pandas scikit-learn joblib plotly numpy
```

Kiểm tra cài thành công:
```bash
python -c "import streamlit, sklearn, joblib, plotly; print('OK')"
```

---

### Bước 4 — Đặt file data đúng chỗ

Đảm bảo file `drone_data_clean.csv` nằm trong thư mục `Data/`:

```
DSS301/
└── Data/
    └── drone_data_clean.csv   ← file này phải có
```

Nếu chưa có thư mục `Data/`, tạo thủ công trong PyCharm:
- Chuột phải vào project → **New → Directory** → đặt tên `Data`
- Kéo file CSV vào thư mục `Data/`

---

### Bước 5 — Train model

Chạy file `train_model.py` để sinh các file `.joblib`:

```bash
python train_model.py
```

Khi chạy xong sẽ thấy:
```
DRONE DSS MODEL TRAINING
Đã đọc data: 200000 dòng, 24 cột
...
Accuracy: 0.9991
...
HOÀN THÀNH TRAIN MODEL
Các file đã được lưu vào thư mục Model/
```

Thư mục `Model/` sẽ xuất hiện với 6 file `.joblib`.

> ⏱️ Thời gian train: khoảng **3–8 phút** tùy cấu hình máy.

---

### Bước 6 — Chạy ứng dụng

Trong Terminal của PyCharm:

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở tại:
```
http://localhost:8501
```

Nếu trình duyệt không tự mở, copy URL trên vào trình duyệt thủ công.

---

## Hướng dẫn sử dụng

### Trang Dashboard

Tổng quan toàn bộ fleet drone:
- **Risk Score tổng hợp** — gauge chart hiển thị mức rủi ro trung bình
- **4 thẻ KPI** — tổng bản ghi, số drone, risk score TB, tỷ lệ High Risk
- **Phân phối Risk Score** — histogram theo từng mức rủi ro
- **Maintenance Action** — bar chart các hành động bảo trì cần thực hiện
- **Battery vs Wind Speed** — scatter plot tương quan pin và gió
- **Risk Score theo Drone** — so sánh rủi ro giữa các drone

### Trang Dự đoán

Nhập thông số cảm biến thực tế để nhận quyết định ngay:

1. Kéo **10 thanh slider** nhập giá trị cảm biến
2. Hệ thống **tự động dự đoán** (không cần nhấn nút)
3. Kết quả hiển thị:
   - Gauge chart **Risk Score ước tính**
   - **Mức rủi ro** + độ tin cậy của model
   - **Hành động bảo trì** cần thực hiện
   - **Tình trạng pin** (Tốt / Trung Bình / Yếu)
   - **Trạng thái bay** (5 cấp độ)
   - **Khuyến nghị** cuối cùng

### Trang Nhập dữ liệu Drone

Nhập số liệu cụ thể (thay vì slider):

1. Điền **Drone ID** (ví dụ: `Drone_Custom_001`)
2. Nhập 10 thông số vào các ô number input
3. Nhấn **Dự đoán**
4. Kết quả được hiển thị và **tự động lưu** vào `Data/custom_drone_data.csv`

### Trang Phân tích Drone

Phân tích chi tiết từng drone:

1. Chọn drone từ dropdown
2. Xem **4 thẻ KPI** của drone đó
3. Xem **histogram Risk Score** của drone
4. Xem **scatter plot** Flight Time vs Battery Level
5. Xem **donut chart** phân bố Maintenance Action

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `No module named 'streamlit'` | Thư viện cài vào Python hệ thống, không vào PyCharm | Vào **Settings → Python Interpreter → Add System Interpreter**, chọn đúng `python.exe` đã cài thư viện |
| `FileNotFoundError: Data/drone_data_clean.csv` | File CSV chưa có trong thư mục `Data/` | Tạo thư mục `Data/` và copy file CSV vào |
| `FileNotFoundError: Model/...joblib` | Chưa chạy `train_model.py` | Chạy `python train_model.py` trước |
| Trình duyệt không tự mở | Streamlit không detect browser | Copy `http://localhost:8501` vào trình duyệt thủ công |
| Train chạy rất chậm | n_jobs=-1 không hoạt động trên một số máy | Thay `n_jobs=-1` thành `n_jobs=1` trong `train_model.py` |

---

## Công nghệ sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `streamlit` | ≥ 1.35 | Web UI framework |
| `scikit-learn` | ≥ 1.3 | Random Forest, LabelEncoder |
| `pandas` | ≥ 2.0 | Xử lý dữ liệu |
| `joblib` | ≥ 1.3 | Lưu/load model |
| `plotly` | ≥ 5.0 | Biểu đồ interactive |
| `numpy` | ≥ 1.24 | Tính toán số học |

---

## Tác giả

**Nhóm DSS301 — FPT University**  
GitHub: [https://github.com/nghiapham179/DSS301](https://github.com/nghiapham179/DSS301)

---

*README này được tạo cho mục đích học thuật — môn DSS301, FPT University.*
