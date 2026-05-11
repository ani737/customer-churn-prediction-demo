import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import io
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = Path("data/ecommerce_customer_churn.csv")
TARGET_COL = "churned"
INPUT_HELP = {
    "age": "Customer age in years.",
    "gender": "Select the gender from the dataset options.",
    "country": "Customer's primary country/region.",
    "tenure_months": "How many months the customer has been active.",
    "avg_order_value": "Average order value in the customer's currency.",
    "orders_last_12m": "Number of orders placed in the last 12 months.",
    "returns_rate": "Return rate as a decimal (0.00 = no returns, 1.00 = all orders returned).",
    "preferred_channel": "Primary shopping channel used most often.",
    "support_tickets": "Support tickets raised in the last 12 months.",
    "has_subscription": "1 = customer has an active subscription, 0 = no subscription.",
    "discount_usage_pct": "Percent of orders using a discount (0-100)."
}
st.title("📊 Customer Churn Prediction Demo")
st.markdown("---")

if "model" not in st.session_state:
    st.session_state.model = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = {}
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []
if "target_col" not in st.session_state:
    st.session_state.target_col = TARGET_COL
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "feature_meta" not in st.session_state:
    st.session_state.feature_meta = {}

@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def preprocess_and_train(df, target_col):
    df = df.copy()
    le = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_col:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            le[col] = encoder
    y = df[target_col]
    if y.dtype == 'object':
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)
        le['_target'] = y_encoder
    drop_cols = [target_col]
    if 'customer_id' in df.columns:
        drop_cols.append('customer_id')
    X = df.drop(columns=drop_cols)
    feature_cols = X.columns.tolist()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, le, feature_cols, scaler, acc, y_test, y_pred


def build_feature_metadata(df, feature_cols):
    meta = {}
    for col in feature_cols:
        series = df[col]
        if series.dtype == 'object':
            options = sorted(series.dropna().astype(str).unique().tolist())
            meta[col] = {
                "type": "categorical",
                "options": options,
                "default": options[0] if options else "",
                "help": INPUT_HELP.get(col, "Select a value from the list.")
            }
        else:
            min_val = float(series.min())
            max_val = float(series.max())
            step = 1 if pd.api.types.is_integer_dtype(series) else 0.01
            default_val = float(series.median())
            if step == 1:
                min_val = int(min_val)
                max_val = int(max_val)
                default_val = int(round(default_val))
            meta[col] = {
                "type": "numeric",
                "min": min_val,
                "max": max_val,
                "step": step,
                "default": default_val,
                "help": INPUT_HELP.get(col, "Enter a numeric value within the dataset range.")
            }
    return meta

def predict_single(model, scaler, feature_cols, label_encoders, input_data):
    df = pd.DataFrame([input_data])
    for col in df.columns:
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = 0
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return pred, proba


if st.session_state.df_raw is None:
    if not DATA_PATH.exists():
        st.error("Dataset not found. Please ensure data/ecommerce_customer_churn.csv exists.")
    else:
        st.session_state.df_raw = load_dataset(DATA_PATH)

st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📤 Upload Dataset", "🧠 Train Model", "🔮 Predict Churn", "📈 Model Performance"]
)

if page == "📤 Upload Dataset":
    st.header("Step 1: Dataset Loaded")
    st.info("This app is hardcoded to use the ecommerce customer churn dataset.")
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.success(f"Dataset ready! Shape: {df.shape}")
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Dataset Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        st.subheader("Column Data Types")
        st.dataframe(pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str)}))
    else:
        st.error("Dataset could not be loaded.")

elif page == "🧠 Train Model":
    st.header("Step 2: Train the Churn Prediction Model")
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw.copy()
        st.write(f"Target column: **{TARGET_COL}**")
        if st.button("🚀 Train Model"):
            with st.spinner("Training model... Please wait."):
                with st.spinner("Preprocessing data and training..."):
                    result = preprocess_and_train(df, TARGET_COL)
                st.session_state.model = result[0]
                st.session_state.label_encoders = result[1]
                st.session_state.feature_cols = result[2]
                st.session_state.scaler = result[3]
                st.session_state.accuracy = result[4]
                st.session_state.y_test = result[5]
                st.session_state.y_pred = result[6]
                st.session_state.feature_meta = build_feature_metadata(df, result[2])
                st.success("Model trained successfully!")
                st.metric("Model Accuracy", f"{result[4]:.2%}")
    else:
        st.warning("Dataset missing. Please ensure the CSV file exists.")

elif page == "🔮 Predict Churn":
    st.header("Step 3: Predict Customer Churn")
    if st.session_state.model is not None:
        st.success("Model is ready!")
        st.subheader("Enter Customer Details")
        inputs = {}
        cols = st.columns(2)
        feature_cols = st.session_state.feature_cols
        feature_meta = st.session_state.feature_meta

        for idx, col in enumerate(feature_cols):
            target_col = cols[idx % 2]
            meta = feature_meta.get(col, {})
            with target_col:
                if col == "has_subscription":
                    inputs[col] = st.selectbox(
                        "has_subscription",
                        options=[0, 1],
                        format_func=lambda val: "Yes" if val == 1 else "No",
                        help=meta.get("help", INPUT_HELP.get(col, "")),
                        key=f"in_{col}"
                    )
                elif col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[col]
                    options = le.classes_.tolist()
                    inputs[col] = st.selectbox(
                        f"{col}",
                        options,
                        help=meta.get("help", INPUT_HELP.get(col, "")),
                        key=f"in_{col}"
                    )
                else:
                    inputs[col] = st.number_input(
                        f"{col}",
                        min_value=meta.get("min", 0.0),
                        max_value=meta.get("max", 0.0),
                        value=meta.get("default", 0.0),
                        step=meta.get("step", 1.0),
                        help=meta.get("help", INPUT_HELP.get(col, "")),
                        key=f"in_{col}"
                    )
        if st.button("🎯 Predict"):
            with st.spinner("Predicting..."):
                pred, proba = predict_single(
                    st.session_state.model, st.session_state.scaler,
                    st.session_state.feature_cols, st.session_state.label_encoders, inputs
                )
                churn_index = list(st.session_state.model.classes_).index(1)
                churn_prob = proba[churn_index]
                not_churn_prob = 1 - churn_prob
                prediction_label = "Churned" if pred == 1 else "Not Churned"

                if pred == 1:
                    st.error(f"**Prediction: {prediction_label}**")
                else:
                    st.success(f"**Prediction: {prediction_label}**")

                st.write(f"**Churn probability:** {churn_prob:.2%}")
                st.write(f"**Not churn probability:** {not_churn_prob:.2%}")

                st.info(
                    "**How to interpret this:**\n"
                    "- The churn probability is the model's confidence that the customer will churn.\n"
                    "- A higher churn probability means the model expects the customer to leave.\n"
                    "- Values are derived from the trained model, not random guesses."
                )
    else:
        st.warning("Please train a model first.")

elif page == "📈 Model Performance":
    st.header("Step 4: Model Performance Metrics")
    if st.session_state.model is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, index=["Actual Not Churned", "Actual Churned"], columns=["Pred Not Churned", "Pred Churned"]))
        with col2:
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            rf = pd.DataFrame(report).transpose()
            st.subheader("Classification Report")
            st.dataframe(rf.round(3))
    else:
        st.warning("Please train a model first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit")
