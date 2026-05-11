import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import io

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
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
    st.session_state.target_col = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

@st.cache_data
def preprocess_and_train(df, target_col):
    df = df.copy()
    le = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            if col != target_col:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                le[col] = encoder
    y = df[target_col]
    if y.dtype == 'object':
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)
        le['_target'] = y_encoder
    X = df.drop(columns=[target_col] + (['customerid'] if 'customerid' in df.columns else []))
    feature_cols = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, le, feature_cols, scaler, acc, y_test, y_pred

def predict_single(model, scaler, feature_cols, label_encoders, input_data):
    df = pd.DataFrame([input_data])
    for col in df.columns:
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except:
                df[col] = 0
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return pred, proba
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📤 Upload Dataset", "🧠 Train Model", "🔮 Predict Churn", "📈 Model Performance"]
)

if page == "📤 Upload Dataset":
    st.header("Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"],
        help="Upload a customer dataset with a target column (e.g., 'churn', 'churned', 'Exited')"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_raw = df
            st.success(f"Dataset loaded! Shape: {df.shape}")
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
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

elif page == "🧠 Train Model":
    st.header("Step 2: Train the Churn Prediction Model")
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw.copy()
        st.subheader("Select Target Column")
        target_options = df.columns.tolist()
        default_target = None
        for col in target_options:
            if col.lower() in ['churn', 'churned', 'exit', 'exited', 'attrition', 'left']:
                default_target = col
                break
        target_col = st.selectbox(
            "Which column is the target (churn)?",
            target_options,
            index=target_options.index(default_target) if default_target else 0
        )
        st.session_state.target_col = target_col
        if st.button("🚀 Train Model"):
            with st.spinner("Training model... Please wait."):
                with st.spinner("Preprocessing data and training..."):
                    result = preprocess_and_train(df, target_col)
                st.session_state.model = result[0]
                st.session_state.label_encoders = result[1]
                st.session_state.feature_cols = result[2]
                st.session_state.scaler = result[3]
                st.session_state.accuracy = result[4]
                st.session_state.y_test = result[5]
                st.session_state.y_pred = result[6]
                st.success("Model trained successfully!")
                st.metric("Model Accuracy", f"{result[4]:.2%}")
    else:
        st.warning("Please upload a dataset first.")

elif page == "🔮 Predict Churn":
    st.header("Step 3: Predict Customer Churn")
    if st.session_state.model is not None:
        st.success("Model is ready!")
        st.subheader("Enter Customer Details")
        num_cols, cat_cols = 8, 8
        nc, cc = st.columns(2)
        inputs = {}
        with nc:
            for col in st.session_state.feature_cols[:num_cols]:
                if col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[col]
                    options = le.classes_.tolist()
                    inputs[col] = st.selectbox(f"{col}", options, key=f"in_{col}")
                else:
                    inputs[col] = st.number_input(f"{col}", value=0.0, key=f"in_{col}")
        with cc:
            for col in st.session_state.feature_cols[num_cols:]:
                if col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[col]
                    options = le.classes_.tolist()
                    inputs[col] = st.selectbox(f"{col}", options, key=f"in_{col}")
                else:
                    inputs[col] = st.number_input(f"{col}", value=0.0, key=f"in_{col}")
        if st.button("🎯 Predict"):
            with st.spinner("Predicting..."):
                pred, proba = predict_single(
                    st.session_state.model, st.session_state.scaler,
                    st.session_state.feature_cols, st.session_state.label_encoders, inputs
                )
                target_encoder = st.session_state.label_encoders.get('_target')
                if target_encoder:
                    labels = target_encoder.inverse_transform([pred, 0])[0]
                    not_label = target_encoder.inverse_transform([0])[0] if pred != 0 else target_encoder.inverse_transform([1])[0]
                    churn_label = target_encoder.inverse_transform([1])[0] if 1 in target_encoder.classes_ else str(1)
                    not_churn_label = target_encoder.inverse_transform([0])[0] if 0 in target_encoder.classes_ else str(0)
                else:
                    churn_label, not_churn_label = "1", "0"
                if pred == (1 if 1 in (target_encoder.classes_ if target_encoder else [0,1]) else 0):
                    st.error(f"**Prediction: CHURN** (Probability: {proba.max():.2%})")
                else:
                    st.success(f"**Prediction: NO CHURN** (Probability: {proba.max():.2%})")
                st.write(f"Churn probability: {proba[1] if len(proba) > 1 else proba[0]:.2%}")
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
            st.dataframe(pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"]))
        with col2:
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            rf = pd.DataFrame(report).transpose()
            st.subheader("Classification Report")
            st.dataframe(rf.round(3))
    else:
        st.warning("Please train a model first.")

st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit")
