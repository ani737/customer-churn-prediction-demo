import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = Path("data/ecommerce_customer_churn.csv")
TARGET_COL = "churned"

FRIENDLY_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "country": "Country",
    "tenure_months": "Tenure (months)",
    "avg_order_value": "Average Order Value",
    "orders_last_12m": "Orders in Last 12 Months",
    "returns_rate": "Returns Rate",
    "preferred_channel": "Preferred Channel",
    "support_tickets": "Support Tickets",
    "has_subscription": "Has Subscription",
    "discount_usage_pct": "Discount Usage Percentage"
}

# Rules for each field: widget type, min, max, step, explanation
FIELD_RULES = {
    "age": {
        "widget": "int",
        "min": 18,
        "max": 100,
        "step": 1,
        "explanation": "Enter the customer's age in whole years. Typical adult range is 18 to 100."
    },
    "tenure_months": {
        "widget": "int",
        "min": 0,
        "max": 240,
        "step": 1,
        "explanation": "Number of months the customer has been active. Use whole numbers only (e.g., 6, 12, 24)."
    },
    "orders_last_12m": {
        "widget": "int",
        "min": 0,
        "max": 500,
        "step": 1,
        "explanation": "Total orders placed in the last 12 months. Whole numbers only (e.g., 5, 10, 25)."
    },
    "support_tickets": {
        "widget": "int",
        "min": 0,
        "max": 100,
        "step": 1,
        "explanation": "Number of support tickets raised in the last 12 months. Whole numbers only (e.g., 0, 3, 10)."
    },
    "avg_order_value": {
        "widget": "float",
        "min": 0.0,
        "max": 100000.0,
        "step": 1.0,
        "explanation": "Average value per order. Enter a realistic currency amount (e.g., 50, 150, 1200)."
    },
    "returns_rate": {
        "widget": "slider_0_1",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "explanation": "Fraction of orders returned. 0.00 = no returns at all, 1.00 = every order was returned. Example: 0.15 means 15% of orders were returned."
    },
    "discount_usage_pct": {
        "widget": "slider_0_100",
        "min": 0,
        "max": 100,
        "step": 1,
        "explanation": "Percentage of orders where a discount was used. Enter a value from 0 to 100. Example: 30 means 30% of orders used a discount."
    },
    "has_subscription": {
        "widget": "binary",
        "explanation": "Does the customer currently have an active subscription? Choose Yes or No."
    },
    "gender": {
        "widget": "categorical",
        "explanation": "Select the customer's gender from the available options in the dataset."
    },
    "country": {
        "widget": "categorical",
        "explanation": "Select the customer's country from the available options. This is a category, not a number."
    },
    "preferred_channel": {
        "widget": "categorical",
        "explanation": "Select the shopping channel the customer uses most often. This is a category, not a number."
    }
}


def init_state():
    defaults = {
        "model": None,
        "df_raw": None,
        "label_encoders": {},
        "feature_cols": [],
        "target_col": TARGET_COL,
        "scaler": None,
        "feature_meta": {},
        "accuracy": None,
        "y_test": None,
        "y_pred": None,
        "feature_importance": None,
        "reference_stats": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def preprocess_and_train(df, target_col):
    df = df.copy()
    le = {}

    for col in df.columns:
        if df[col].dtype == "object" and col != target_col:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            le[col] = encoder

    y = df[target_col]
    if y.dtype == "object":
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))
        le["_target"] = y_encoder

    drop_cols = [target_col]
    if "customer_id" in df.columns:
        drop_cols.append("customer_id")

    X = df.drop(columns=drop_cols)
    feature_cols = X.columns.tolist()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    importance = dict(zip(feature_cols, model.feature_importances_))

    reference_stats = {}
    raw_X = df.drop(columns=drop_cols, errors="ignore")
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(raw_X[col]):
            reference_stats[col] = {
                "median": float(pd.to_numeric(raw_X[col], errors="coerce").median()),
                "q1": float(pd.to_numeric(raw_X[col], errors="coerce").quantile(0.25)),
                "q3": float(pd.to_numeric(raw_X[col], errors="coerce").quantile(0.75)),
                "mode": None
            }
        else:
            reference_stats[col] = {
                "median": None,
                "q1": None,
                "q3": None,
                "mode": raw_X[col].mode().iloc[0] if len(raw_X[col].mode()) > 0 else None
            }

    return model, le, feature_cols, scaler, acc, y_test, y_pred, importance, reference_stats


def build_feature_metadata(df, feature_cols):
    meta = {}

    for col in feature_cols:
        series = df[col]
        rule = FIELD_RULES.get(col, {})

        if col in FIELD_RULES and FIELD_RULES[col]["widget"] == "categorical":
            options = sorted(series.dropna().astype(str).unique().tolist())
            meta[col] = {
                "type": "categorical",
                "options": options,
                "default": options[0] if options else "",
                "help": FIELD_RULES[col]["explanation"]
            }
            continue

        if col == "has_subscription":
            meta[col] = {
                "type": "binary",
                "options": [1, 0],
                "default": 1,
                "help": FIELD_RULES[col]["explanation"]
            }
            continue

        if pd.api.types.is_numeric_dtype(series):
            data_min = float(pd.to_numeric(series, errors="coerce").min())
            data_max = float(pd.to_numeric(series, errors="coerce").max())
            data_median = float(pd.to_numeric(series, errors="coerce").median())

            min_val = data_min
            max_val = data_max
            widget = rule.get("widget", "float")
            step = rule.get("step", 1.0)

            if "min" in rule:
                min_val = max(min_val, float(rule["min"]))
            if "max" in rule:
                max_val = min(max_val, float(rule["max"]))

            default_val = min(max(data_median, min_val), max_val)

            if widget in ["int", "slider_0_100"]:
                min_val = int(round(min_val))
                max_val = int(round(max_val))
                default_val = int(round(default_val))

            meta[col] = {
                "type": "numeric",
                "widget": widget,
                "min": min_val,
                "max": max_val,
                "step": step,
                "default": default_val,
                "help": rule.get("explanation", f"Enter a valid value for {col}."),
                "range_text": f"Dataset range: {data_min:.2f} to {data_max:.2f}"
            }

    return meta


def predict_single(model, scaler, feature_cols, label_encoders, input_data):
    df = pd.DataFrame([input_data])

    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col].astype(str))

    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return pred, proba


def get_prediction_label(pred):
    target_encoder = st.session_state.label_encoders.get("_target")
    if target_encoder is not None:
        return str(target_encoder.inverse_transform([pred])[0])
    return "Churned" if pred == 1 else "Not Churned"


def get_risk_band(churn_probability):
    if churn_probability < 0.35:
        return "Low"
    if churn_probability < 0.65:
        return "Medium"
    return "High"


def explain_prediction(inputs, churn_probability):
    importance = st.session_state.feature_importance or {}
    reference_stats = st.session_state.reference_stats or {}
    reasons = []

    for feature, value in inputs.items():
        weight = importance.get(feature, 0)
        stats = reference_stats.get(feature, {})

        if isinstance(value, (int, float)) and stats.get("median") is not None:
            median = stats["median"]
            q1 = stats.get("q1")
            q3 = stats.get("q3")

            label = FRIENDLY_LABELS.get(feature, feature)

            if feature in ["support_tickets"] and value > median:
                reasons.append((weight, f"{label} is above the typical level ({value} vs median {median:.1f}), suggesting the customer may be frustrated."))
            elif feature in ["support_tickets"] and value == 0:
                reasons.append((weight - 0.05, f"{label} is zero, which is a positive sign of a smooth customer experience."))
            elif feature in ["returns_rate"] and value > median:
                reasons.append((weight, f"{label} is above the typical level ({value:.2f} vs median {median:.2f}), which can indicate dissatisfaction."))
            elif feature in ["returns_rate"] and value < 0.1:
                reasons.append((weight - 0.03, f"{label} is very low ({value:.2f}), suggesting the customer is generally satisfied with purchases."))
            elif feature in ["tenure_months"] and value < median:
                reasons.append((weight, f"{label} is below the typical level ({value} vs median {median:.1f} months). Newer customers tend to churn more often."))
            elif feature in ["tenure_months"] and value > median * 1.5:
                reasons.append((weight - 0.04, f"{label} is well above average ({value} months), suggesting a loyal, long-term customer."))
            elif feature in ["orders_last_12m"] and value < median:
                reasons.append((weight, f"{label} is lower than typical activity ({value} vs median {median:.1f}), which may signal disengagement."))
            elif feature in ["orders_last_12m"] and value > median * 1.5:
                reasons.append((weight - 0.05, f"{label} is significantly above average ({value} orders), indicating an active, engaged customer."))
            elif feature in ["discount_usage_pct"] and value > 70:
                reasons.append((weight, f"{label} is very high ({value}%), which may indicate price sensitivity and higher churn risk."))
            elif feature in ["discount_usage_pct"] and value < 2:
                reasons.append((weight - 0.03, f"{label} is very low ({value}%), suggesting the customer does not rely heavily on discounts."))
            elif feature in ["avg_order_value"] and q1 is not None and value < q1:
                reasons.append((weight, f"{label} is in the lower range of the dataset ({value} vs lower quartile {q1:.1f}), which may indicate a budget-conscious customer."))
            elif feature in ["avg_order_value"] and q3 is not None and value > q3:
                reasons.append((weight - 0.02, f"{label} is in the higher range of the dataset ({value} vs upper quartile {q3:.1f}), suggesting a premium customer."))

        if feature == "has_subscription":
            label = FRIENDLY_LABELS.get(feature, feature)
            if int(value) == 0:
                reasons.append((weight + 0.05, f"The customer does not have an active subscription, which is a common factor in churn cases."))
            else:
                reasons.append((weight - 0.04, f"The customer has an active subscription, which is a strong retention signal."))

        if feature == "preferred_channel":
            label = FRIENDLY_LABELS.get(feature, feature)
            if str(value).lower() in ["unknown", "other", ""]:
                reasons.append((weight, f"{label} is not well-defined, which may indicate weaker engagement patterns."))
            else:
                reasons.append((weight - 0.02, f"The customer has a clear {label} preference, indicating established shopping habits."))

    reasons = sorted(reasons, key=lambda x: x[0], reverse=True)
    top_reasons = [text for _, text in reasons[:5]]

    if not top_reasons:
        if churn_probability >= 0.5:
            top_reasons = ["The input values align with patterns the model has learned from past churn cases."]
        else:
            top_reasons = ["Most input values match stable customer patterns seen in the training data."]

    return top_reasons


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

st.title("📊 Customer Churn Prediction Demo")
st.markdown("---")

if page == "📤 Upload Dataset":
    st.header("Step 1: Dataset Loaded")
    st.info("This app uses the ecommerce customer churn dataset from the local data folder.")

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.success(f"Dataset ready! Shape: {df.shape}")
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

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
            with st.spinner("Training model..."):
                result = preprocess_and_train(df, TARGET_COL)

            st.session_state.model = result[0]
            st.session_state.label_encoders = result[1]
            st.session_state.feature_cols = result[2]
            st.session_state.scaler = result[3]
            st.session_state.accuracy = result[4]
            st.session_state.y_test = result[5]
            st.session_state.y_pred = result[6]
            st.session_state.feature_importance = result[7]
            st.session_state.reference_stats = result[8]
            st.session_state.feature_meta = build_feature_metadata(df, result[2])

            st.success("Model trained successfully!")
            st.metric("Model Accuracy", f"{result[4]:.2%}")

            importance_df = pd.DataFrame({
                "Feature": list(st.session_state.feature_importance.keys()),
                "Importance": list(st.session_state.feature_importance.values())
            }).sort_values("Importance", ascending=False)

            st.subheader("Top Important Features")
            st.dataframe(importance_df.head(10), use_container_width=True)
    else:
        st.warning("Dataset missing. Please ensure the CSV file exists.")

elif page == "🔮 Predict Churn":
    st.header("Step 3: Predict Customer Churn")

    if st.session_state.model is not None:
        st.success("Model is ready.")
        st.subheader("Enter Customer Details")
        st.caption("Each field shows the expected range. Only realistic values are accepted.")

        inputs = {}
        cols = st.columns(2)
        feature_cols = st.session_state.feature_cols
        feature_meta = st.session_state.feature_meta

        for idx, col in enumerate(feature_cols):
            meta = feature_meta.get(col, {})
            box = cols[idx % 2]

            with box:
                label = FRIENDLY_LABELS.get(col, col)
                help_text = meta.get("help", "")
                range_text = meta.get("range_text", "")
                if range_text:
                    st.caption(range_text)

                if meta.get("type") == "binary":
                    value = st.selectbox(
                        label,
                        options=[1, 0],
                        format_func=lambda v: "Yes" if v == 1 else "No",
                        help=help_text,
                        key=f"in_{col}"
                    )
                    inputs[col] = int(value)

                elif meta.get("type") == "categorical" or col in st.session_state.label_encoders:
                    options = meta.get("options")
                    if not options and col in st.session_state.label_encoders:
                        options = st.session_state.label_encoders[col].classes_.tolist()

                    inputs[col] = st.selectbox(
                        label,
                        options=options,
                        help=help_text,
                        key=f"in_{col}"
                    )

                elif meta.get("type") == "numeric":
                    widget = meta.get("widget", "float")
                    if widget == "int":
                        inputs[col] = st.number_input(
                            label,
                            min_value=int(meta["min"]),
                            max_value=int(meta["max"]),
                            value=int(meta["default"]),
                            step=int(meta["step"]),
                            help=help_text,
                            key=f"in_{col}"
                        )
                    elif widget == "slider_0_1":
                        inputs[col] = st.slider(
                            label,
                            min_value=float(meta["min"]),
                            max_value=float(meta["max"]),
                            value=float(meta["default"]),
                            step=float(meta["step"]),
                            help=help_text,
                            key=f"in_{col}"
                        )
                    elif widget == "slider_0_100":
                        inputs[col] = st.slider(
                            label,
                            min_value=int(meta["min"]),
                            max_value=int(meta["max"]),
                            value=int(meta["default"]),
                            step=int(meta["step"]),
                            help=help_text,
                            key=f"in_{col}"
                        )
                    else:
                        inputs[col] = st.number_input(
                            label,
                            min_value=float(meta["min"]),
                            max_value=float(meta["max"]),
                            value=float(meta["default"]),
                            step=float(meta["step"]),
                            help=help_text,
                            key=f"in_{col}"
                        )

        if st.button("🔍 Predict Churn"):
            pred, proba = predict_single(
                st.session_state.model,
                st.session_state.scaler,
                st.session_state.feature_cols,
                st.session_state.label_encoders,
                inputs
            )

            churn_probability = float(np.max(proba))
            label = get_prediction_label(pred)
            risk_band = get_risk_band(churn_probability)

            st.subheader("Prediction Result")

            if risk_band == "Low":
                st.success(f"**Prediction: {label}**")
            elif risk_band == "Medium":
                st.warning(f"**Prediction: {label}**")
            else:
                st.error(f"**Prediction: {label}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{churn_probability:.2%}")
            col2.metric("Risk Band", risk_band)
            col3.metric("Model", "Random Forest")

            st.progress(churn_probability)

            st.subheader("Why This Prediction?")
            st.caption("The model compares your inputs against patterns learned from training data.")
            reasons = explain_prediction(inputs, churn_probability)
            for i, reason in enumerate(reasons, 1):
                st.write(f"**{i}.** {reason}")

            st.info(
                "This result is based on patterns the model learned from historical customer data. "
                "The confidence score reflects the probability that the customer will churn. "
                "The explanation highlights which of your input values are driving the prediction."
            )
    else:
        st.warning("Please train the model first.")

elif page == "📈 Model Performance":
    st.header("Step 4: Model Performance Metrics")

    if st.session_state.model is not None and st.session_state.y_test is not None:
        st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")

        report = classification_report(
            st.session_state.y_test,
            st.session_state.y_pred,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)

        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        cm_df = pd.DataFrame(cm)
        st.subheader("Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

        importance_df = pd.DataFrame({
            "Feature": list(st.session_state.feature_importance.keys()),
            "Importance": list(st.session_state.feature_importance.values())
        }).sort_values("Importance", ascending=False)

        st.subheader("Feature Importance")
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.warning("Please train the model first.")
