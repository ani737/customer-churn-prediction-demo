# Customer Churn Prediction Demo

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

A **Streamlit web application** that allows you to upload a customer dataset, train a machine learning model, and predict customer churn — all through an interactive browser interface.

---

## Features

- **CSV Upload**: Upload your own customer churn dataset
- **Auto Target Detection**: Automatically detects common churn target columns
- **On-the-fly Training**: Trains a Random Forest model using your data
- **Model Evaluation**: View accuracy, confusion matrix, and classification report
- **Single Prediction**: Predict churn for individual customers interactively
- **Session Persistence**: Model stays in memory during your session

---

## Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn (Random Forest Classifier)
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Model Persistence**: joblib

---

## Installation

### Clone the repository
```bash
git clone https://github.com/ani737/customer-churn-prediction-demo.git
cd customer-churn-prediction-demo
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
streamlit run app.py
```

---

## How to Use

### Step 1: Upload Dataset
Navigate to the **Upload Dataset** tab and upload a CSV file containing customer data. The dataset should have a target column indicating churn (e.g., `churn`, `churned`, `exited`, `attrition`).

### Step 2: Train Model
Go to the **Train Model** tab, select your target column, and click **Train Model**. The app will preprocess the data and train a Random Forest classifier.

### Step 3: Predict Churn
In the **Predict Churn** tab, enter customer details through interactive dropdowns and number inputs, then click **Predict** to get a churn prediction.

### Step 4: View Performance
Check the **Model Performance** tab to see accuracy metrics, confusion matrix, and classification report.

---

## Dataset Format

Your CSV file should contain:
- **Feature columns**: Any customer attributes (numerical or categorical)
- **Target column**: Binary churn indicator (e.g., 0/1, Yes/No, True/False)

The app automatically:
- Encodes categorical features using Label Encoding
- Scales numerical features using StandardScaler
- Handles missing values gracefully

---

## Project Structure

```
customer-churn-prediction-demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

---

## Deployment

### Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and deploy
5. Enter `app.py` as the main file

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---

## License

This project is licensed under the MIT License.

---

Made with ❤️ by **ani737**
