# Drone DSS - Decision Support System for Drone Operation and Maintenance

## 1. Project Overview

**Drone DSS** is a Decision Support System prototype designed to support **drone operation and maintenance decisions**.

The system analyzes drone operational data such as battery level, flight time, signal strength, wind speed, GPS accuracy, altitude, speed, temperature, humidity, and pressure. Based on these inputs, the system predicts:

- Operation risk level
- Maintenance action
- Battery status
- Flight status
- Final recommendation for the operator

This project is built for the **DSS301** course and uses **Python**, **Streamlit**, **Scikit-learn**, **Random Forest Classifier**, and **Plotly**.

---

## 2. Main Features

### 2.1 Dashboard

The dashboard provides an overview of the drone dataset, including:

- Total number of records
- Number of drones
- Average risk score
- Percentage of high-risk cases
- Risk score distribution
- Maintenance action distribution
- Battery level versus wind speed analysis
- Average risk score by drone

### 2.2 Prediction Page

The prediction page allows users to adjust drone operation parameters using sliders.

The system then predicts:

- Operation risk
- Maintenance action
- Battery status
- Flight status
- Final recommendation

### 2.3 Manual Drone Data Input

The manual input page allows operators to enter drone data directly through a form.

After clicking **Predict**, the system will:

1. Predict the operation risk.
2. Predict the maintenance action.
3. Determine the battery status.
4. Determine the flight status.
5. Generate the final recommendation.
6. Save the input and prediction result into:

```text
Data/custom_drone_data.csv
```

### 2.4 Drone Analysis

The drone analysis page allows users to select a specific drone and view:

- Number of records
- Average risk score
- High-risk percentage
- Average battery level
- Risk score distribution
- Flight time versus battery level
- Maintenance action distribution

---

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Pandas | Data processing |
| NumPy | Numerical processing |
| Scikit-learn | Machine learning model training |
| Random Forest Classifier | Prediction model |
| Joblib | Saving and loading trained models |
| Plotly | Interactive charts and visualizations |
| OpenPyXL | Excel file support |

---

## 4. Project Structure

```text
DSS301/
│
├── Data/
│   ├── drone_data.csv
│   ├── drone_data_clean.csv
│   └── custom_drone_data.csv
│
├── Model/
│   ├── operation_risk_model.joblib
│   ├── operation_risk_model_label_encoder.joblib
│   ├── maintenance_action_model.joblib
│   ├── maintenance_action_model_label_encoder.joblib
│   ├── recommendation_model.joblib
│   └── recommendation_model_label_encoder.joblib
│
├── app.py
├── train_model.py
├── ui.py
├── Requirement.txt
└── README.md
```

### Folder and File Explanation

| File / Folder | Description |
|---|---|
| `Data/` | Contains the drone datasets |
| `Data/drone_data_clean.csv` | Main cleaned dataset used by the system |
| `Data/custom_drone_data.csv` | Stores manually entered drone data and prediction results |
| `Model/` | Stores trained machine learning models |
| `app.py` | Main Streamlit application |
| `train_model.py` | Trains and saves machine learning models |
| `ui.py` | Contains UI styling and reusable UI components |
| `Requirement.txt` | Lists required Python libraries |
| `README.md` | Project setup and usage guide |

---

## 5. Requirements

Before running the project, make sure the following software is installed:

- Python 3.10 or later
- PyCharm or Visual Studio Code
- Google Chrome, Microsoft Edge, or Firefox
- Git, if cloning the project from GitHub

Required Python libraries:

```text
pandas
numpy
scikit-learn
joblib
streamlit
plotly
openpyxl
```

---

## 6. How to Download the Project

### Option 1: Download ZIP

1. Open the GitHub repository.
2. Click the **Code** button.
3. Select **Download ZIP**.
4. Extract the ZIP file.
5. Open the project folder using PyCharm or Visual Studio Code.

### Option 2: Clone with Git

Open Terminal, Command Prompt, PowerShell, or Git Bash and run:

```bash
git clone <your-github-repository-link>
```

Then move into the project folder:

```bash
cd DSS301
```

---

## 7. Installation

Open the terminal inside the project folder and install the required libraries.

### Install directly

```bash
python -m pip install pandas numpy scikit-learn joblib streamlit plotly openpyxl
```

### Or install from `Requirement.txt`

```bash
python -m pip install -r Requirement.txt
```

Recommended `Requirement.txt` content:

```text
pandas
numpy
scikit-learn
joblib
streamlit
plotly
openpyxl
```

---

## 8. Train the Models

Before running the application, train the models by running:

```bash
python train_model.py
```

After training, the following files should appear in the `Model/` folder:

```text
operation_risk_model.joblib
operation_risk_model_label_encoder.joblib
maintenance_action_model.joblib
maintenance_action_model_label_encoder.joblib
recommendation_model.joblib
recommendation_model_label_encoder.joblib
```

If these files already exist, you can skip this step unless the dataset or training logic has changed.

---

## 9. Run the Application

Run the Streamlit app using:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

If it does not open automatically, copy the local URL from the terminal, usually:

```text
http://localhost:8501
```

Then paste it into your browser.

---

## 10. Model Explanation

The project uses three machine learning models.

| Model | Output | Purpose |
|---|---|---|
| `operation_risk_model` | `operation_risk` | Predicts the operational risk level |
| `maintenance_action_model` | `maintenance_action` | Predicts the required maintenance action |
| `recommendation_model` | `recommendation` | Provides the final recommendation |

The models use the following 10 input features:

```text
battery_level
flight_time
signal_strength
temperature
wind_speed
gps_accuracy
altitude
speed
humidity
pressure
```

---

## 11. Flight Status Logic

In addition to machine learning predictions, the system uses a rule-based decision layer to make the result easier for operators to understand.

### Flight Status Categories

| Flight Status | Meaning |
|---|---|
| `Đủ Điều Kiện Bay` | The drone is safe to fly |
| `Bay Kèm Giám Sát` | The drone can fly but requires close monitoring |
| `Quay Về Trạm` | The drone should return to base |
| `Cấm Bay` | The drone should not fly |
| `Yêu Cầu Bảo Trì` | The drone requires inspection or maintenance |

### Risk Conditions

| Parameter | Warning / Dangerous Condition |
|---|---|
| Battery level | Below 20% is dangerous |
| Signal strength | Below 35% is dangerous |
| Wind speed | Above 35 is dangerous |
| GPS accuracy | Above 10 is dangerous |
| Temperature | Above 45°C or below -10°C is dangerous |
| Flight time | Above 40 minutes requires monitoring |
| Altitude | Above 350 meters requires monitoring |
| Speed | Above 75 requires monitoring |

---

## 12. Custom Data Saving

When users enter drone data manually and click **Predict**, the system saves the input and prediction result into:

```text
Data/custom_drone_data.csv
```

The saved file includes:

```text
drone_id
battery_level
flight_time
signal_strength
temperature
wind_speed
gps_accuracy
altitude
speed
humidity
pressure
operation_risk
maintenance_action
recommendation
battery_status
flight_status
flight_reason
created_at
```

Each new prediction is appended as a new row.

---

## 13. Common Errors and Fixes

### Error 1: `streamlit is not recognized`

Use this command instead:

```bash
python -m streamlit run app.py
```

If Streamlit is not installed, run:

```bash
python -m pip install streamlit
```

### Error 2: Missing model files

If the app cannot find `.joblib` files, run:

```bash
python train_model.py
```

Then run the app again:

```bash
python -m streamlit run app.py
```

### Error 3: Dataset not found

Make sure the dataset is located at:

```text
Data/drone_data_clean.csv
```

The folder name must be `Data`, not `data`.

### Error 4: `pandas.errors.EmptyDataError: No columns to parse from file`

This may happen when `Data/custom_drone_data.csv` exists but is empty.

Quick fix:

1. Delete `Data/custom_drone_data.csv`.
2. Run the app again.
3. Submit a new manual prediction.

Recommended code fix:

Make sure the `save_custom_drone_data()` function handles empty files before reading the CSV.

---

## 14. Recommended Running Order

### First-time setup

```bash
python -m pip install -r Requirement.txt
python train_model.py
python -m streamlit run app.py
```

### Later runs

If models are already trained:

```bash
python -m streamlit run app.py
```

Only retrain the models when the dataset or training code changes.

---

## 15. Notes

This project is a prototype for the DSS301 course.

The goal is not to fully automate drone operation, but to support human decision-making by providing:

- Risk prediction
- Maintenance suggestion
- Flight status assessment
- Final operational recommendation

The system helps drone operators make faster, more consistent, and more explainable decisions during drone operation and maintenance.
