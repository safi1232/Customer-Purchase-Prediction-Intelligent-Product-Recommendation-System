import time
from datetime import datetime

import requests
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Purchase & Recommendation",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_OPTIONS = [
    "Best Model",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "All Models"
]

MONTH_OPTIONS = [
    "Jan", "Feb", "Mar", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

VISITOR_TYPE_OPTIONS = [
    "Returning_Visitor",
    "New_Visitor",
    "Other"
]

REQUEST_TIMEOUT = 10
MAX_HEALTH_HISTORY = 20


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://127.0.0.1:8000"

if "health_status" not in st.session_state:
    st.session_state.health_status = None

if "health_history" not in st.session_state:
    st.session_state.health_history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None

if "models_info" not in st.session_state:
    st.session_state.models_info = None


# ============================================================
# HELPERS
# ============================================================

def call_api(method, path, json_body=None):
    """
    Wraps a call to the backend API.
    Returns (success, payload, status_code, error_message, latency_ms)
    """

    url = st.session_state.api_base_url.rstrip("/") + path

    start = time.perf_counter()

    try:

        if method == "GET":
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        else:
            response = requests.post(url, json=json_body, timeout=REQUEST_TIMEOUT)

    except requests.exceptions.ConnectionError:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return False, None, None, (
            "Could not connect to the API. Is the backend running at "
            f"`{st.session_state.api_base_url}`?"
        ), latency_ms

    except requests.exceptions.Timeout:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return False, None, None, "The request timed out.", latency_ms

    except requests.exceptions.RequestException as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return False, None, None, f"Request failed: {str(e)}", latency_ms

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    if response.status_code == 200:
        try:
            return True, response.json(), response.status_code, None, latency_ms
        except ValueError:
            return False, None, response.status_code, "API returned an invalid JSON response.", latency_ms

    try:
        error_body = response.json()
        detail = error_body.get("detail", response.text)
    except ValueError:
        detail = response.text

    return False, None, response.status_code, detail, latency_ms


def format_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def record_health_check():

    success, payload, status_code, error, latency_ms = call_api("GET", "/health")

    entry = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Status": "🟢 Online" if success and payload.get("model_loaded") else "🔴 Offline",
        "Latency (ms)": latency_ms,
        "Version": payload.get("version") if success else "N/A"
    }

    st.session_state.health_history.insert(0, entry)
    st.session_state.health_history = st.session_state.health_history[:MAX_HEALTH_HISTORY]

    if success:
        st.session_state.health_status = payload
    else:
        st.session_state.health_status = {"error": error}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛒 Purchase & Recs")

    st.divider()

    st.subheader("API Configuration")

    st.session_state.api_base_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL of the FastAPI backend"
    )

    if st.button("Check API Health", use_container_width=True):
        record_health_check()

    if st.session_state.health_status is not None:

        health = st.session_state.health_status

        if "error" in health:
            st.error(f"API unreachable: {health['error']}")
        else:
            if health.get("status") == "ok" and health.get("model_loaded"):
                st.success(f"API online — v{health.get('version', '?')}")
            else:
                st.warning(f"API responded but model may not be loaded (v{health.get('version', '?')})")

    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "Purchase Prediction",
            "Product Recommendations",
            "Model Health Tracker",
            "Model Performance Tracker"
        ],
        index=0
    )

    st.divider()

    with st.expander("ℹ️ About this app"):
        st.write(
            "This app talks to a FastAPI backend that predicts whether an "
            "online shopping session will end in a purchase, using several "
            "trained ML models. It also recommends similar products and "
            "tracks API health and model performance over time."
        )


# ============================================================
# PAGE 1 — PURCHASE PREDICTION
# ============================================================

if page == "Purchase Prediction":

    st.title("🛍️ Purchase Prediction")
    st.caption("Predict whether a shopping session will result in a purchase.")

    st.divider()

    with st.form("prediction_form"):

        with st.expander("📊 Session Behavior", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                administrative = st.number_input(
                    "Administrative Pages",
                    min_value=0, step=1, value=2
                )
                administrative_duration = st.number_input(
                    "Administrative Duration (sec)",
                    min_value=0.0, step=1.0, value=50.5
                )

            with col2:
                informational = st.number_input(
                    "Informational Pages",
                    min_value=0, step=1, value=1
                )
                informational_duration = st.number_input(
                    "Informational Duration (sec)",
                    min_value=0.0, step=1.0, value=20.0
                )

            with col3:
                product_related = st.number_input(
                    "Product-Related Pages",
                    min_value=0, step=1, value=30
                )
                product_related_duration = st.number_input(
                    "Product-Related Duration (sec)",
                    min_value=0.0, step=1.0, value=800.5
                )

        with st.expander("📈 Engagement Metrics", expanded=True):

            col1, col2 = st.columns(2)

            with col1:
                bounce_rates = st.slider(
                    "Bounce Rate",
                    min_value=0.0, max_value=1.0, value=0.02, step=0.01
                )
                exit_rates = st.slider(
                    "Exit Rate",
                    min_value=0.0, max_value=1.0, value=0.04, step=0.01
                )

            with col2:
                page_values = st.number_input(
                    "Page Values",
                    min_value=0.0, step=0.5, value=5.5,
                    help="Average value of pages visited before completing a transaction"
                )
                special_day = st.slider(
                    "Special Day Closeness",
                    min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                    help="Closeness of the visit date to a special day (e.g. Valentine's Day)"
                )

        with st.expander("🌐 Visit Context", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                month = st.selectbox("Month", options=MONTH_OPTIONS, index=MONTH_OPTIONS.index("May"))
                visitor_type = st.selectbox(
                    "Visitor Type",
                    options=VISITOR_TYPE_OPTIONS,
                    index=VISITOR_TYPE_OPTIONS.index("Returning_Visitor")
                )

            with col2:
                operating_systems = st.number_input("Operating System (code)", min_value=0, step=1, value=2)
                browser = st.number_input("Browser (code)", min_value=0, step=1, value=1)

            with col3:
                region = st.number_input("Region (code)", min_value=0, step=1, value=1)
                traffic_type = st.number_input("Traffic Type (code)", min_value=0, step=1, value=2)

            weekend = st.checkbox("Visit occurred on a weekend", value=False)

        with st.expander("🤖 Model Selection", expanded=True):

            model_name = st.selectbox(
                "Model",
                options=MODEL_OPTIONS,
                index=0,
                help="Choose a specific model, or 'All Models' to compare every trained model at once."
            )

        st.divider()

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

    if submitted:

        payload = {
            "Administrative": int(administrative),
            "Administrative_Duration": float(administrative_duration),
            "Informational": int(informational),
            "Informational_Duration": float(informational_duration),
            "ProductRelated": int(product_related),
            "ProductRelated_Duration": float(product_related_duration),
            "BounceRates": float(bounce_rates),
            "ExitRates": float(exit_rates),
            "PageValues": float(page_values),
            "SpecialDay": float(special_day),
            "Month": month,
            "OperatingSystems": int(operating_systems),
            "Browser": int(browser),
            "Region": int(region),
            "TrafficType": int(traffic_type),
            "VisitorType": visitor_type,
            "Weekend": bool(weekend),
            "model_name": model_name
        }

        with st.spinner("Running prediction..."):
            success, result, status_code, error, _ = call_api("POST", "/predict", json_body=payload)

        if success:
            st.session_state.prediction_result = result
        else:
            st.session_state.prediction_result = None

            if status_code == 422:
                st.error(f"Validation error: {error}")
            elif status_code == 500:
                st.error(f"Server error: {error}")
            else:
                st.error(f"Prediction failed: {error}")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if st.session_state.prediction_result is not None:

        result = st.session_state.prediction_result

        st.divider()
        st.subheader("Result")

        will_purchase = result.get("prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            if will_purchase:
                st.success("### ✅ Will Purchase")
            else:
                st.error("### ❌ Will Not Purchase")

        with col2:
            st.metric("Purchase Probability", format_pct(result.get("purchase_probability")))

        with col3:
            st.metric("Confidence", format_pct(result.get("confidence")))

        st.caption(f"Model used: **{result.get('selected_model')}**")

        model_results = result.get("model_results") or []

        if model_results:

            st.divider()

            if len(model_results) > 1:
                st.subheader(f"Model Comparison ({len(model_results)} models)")
            else:
                st.subheader("Model Result")
                st.info(
                    "Only one model was returned. To compare all models, select "
                    "**'All Models'** in Model Selection above — this requires the "
                    "backend's /predict endpoint to support per-model routing."
                )

            df = pd.DataFrame(model_results)

            df = df.rename(columns={
                "model": "Model",
                "prediction": "Prediction",
                "purchase_probability": "Purchase Probability",
                "confidence": "Confidence",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1 Score"
            })

            df = df.sort_values(by="F1 Score", ascending=False, na_position="last").reset_index(drop=True)

            display_df = df.copy()

            for col in ["Purchase Probability", "Confidence", "Accuracy", "Precision", "Recall", "F1 Score"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
                    )

            display_df["Prediction"] = display_df["Prediction"].apply(
                lambda v: "Purchase" if v else "No Purchase"
            )

            selected_model_name = result.get("selected_model")

            def highlight_selected(row):
                if row["Model"] == selected_model_name:
                    return ["background-color: #e6f4ea"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_df.style.apply(highlight_selected, axis=1),
                use_container_width=True,
                hide_index=True
            )

            if len(df) > 1:

                metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score"] if c in df.columns]

                chart_df = df[["Model"] + metric_cols].dropna(how="all", subset=metric_cols).set_index("Model")

                if not chart_df.empty:
                    st.caption("Accuracy / Precision / Recall / F1 Score by Model")
                    st.bar_chart(chart_df)

        recommendations = result.get("recommendations") or []

        if recommendations:
            st.divider()
            st.subheader("Related Recommendations")
            st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2 — PRODUCT RECOMMENDATIONS
# ============================================================

elif page == "Product Recommendations":

    st.title("🎁 Product Recommendations")
    st.caption("Find products similar to a given stock code, based on customer purchase patterns.")

    st.divider()

    with st.form("recommendation_form"):

        col1, col2 = st.columns([3, 1])

        with col1:
            stock_code = st.text_input(
                "Stock Code",
                placeholder="e.g. 85123A",
                help="The product's stock/SKU code"
            )

        with col2:
            n = st.number_input(
                "Number of Recommendations",
                min_value=1, max_value=50, value=10, step=1
            )

        submitted = st.form_submit_button("🔍 Get Recommendations", use_container_width=True, type="primary")

    if submitted:

        if not stock_code.strip():

            st.warning("Please enter a stock code.")

        else:

            payload = {
                "stock_code": stock_code.strip(),
                "n": int(n)
            }

            with st.spinner("Finding similar products..."):
                success, result, status_code, error, _ = call_api("POST", "/recommend", json_body=payload)

            if success:
                st.session_state.recommendation_result = result
            else:
                st.session_state.recommendation_result = None

                if status_code == 404:
                    st.warning(f"⚠️ {error}")
                elif status_code == 422:
                    st.error(f"Validation error: {error}")
                else:
                    st.error(f"Recommendation failed: {error}")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if st.session_state.recommendation_result is not None:

        result = st.session_state.recommendation_result
        recommendations = result.get("recommendations") or []

        st.divider()
        st.subheader(f"Results for '{result.get('stock_code')}'")
        st.caption(f"{result.get('number_of_recommendations', len(recommendations))} recommendations found")

        if recommendations:

            df = pd.DataFrame(recommendations)

            if "Similarity" in df.columns:
                df = df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)

            display_df = df.copy()

            if "Similarity" in display_df.columns:
                display_df["Similarity"] = display_df["Similarity"].apply(
                    lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "N/A"
                )

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if "Similarity" in df.columns:

                chart_label_col = "Description" if "Description" in df.columns else "StockCode"

                chart_df = df[[chart_label_col, "Similarity"]].dropna().set_index(chart_label_col)

                if not chart_df.empty:
                    st.caption("Similarity by Product")
                    st.bar_chart(chart_df)

        else:
            st.info("No recommendations returned.")


# ============================================================
# PAGE 3 — MODEL HEALTH TRACKER
# ============================================================

elif page == "Model Health Tracker":

    st.title("🩺 Model Health Tracker")
    st.caption("Live status checks of the backend API and model availability over time.")

    st.divider()

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔄 Run Health Check Now", use_container_width=True, type="primary"):
            record_health_check()

    with col2:
        if st.session_state.health_status and "error" not in st.session_state.health_status:
            h = st.session_state.health_status
            st.success(
                f"Status: **{h.get('status')}**  |  "
                f"Model Loaded: **{h.get('model_loaded')}**  |  "
                f"Version: **{h.get('version')}**"
            )
        elif st.session_state.health_status:
            st.error(f"Last check failed: {st.session_state.health_status.get('error')}")
        else:
            st.info("No health checks recorded yet. Click the button to run one.")

    st.divider()

    if st.session_state.health_history:

        st.subheader("Check History")

        history_df = pd.DataFrame(st.session_state.health_history)

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        latency_df = history_df[["Time", "Latency (ms)"]].set_index("Time")

        if not latency_df.empty:
            st.caption("Response Latency Over Recent Checks (oldest to newest)")
            st.line_chart(latency_df.iloc[::-1])

        online_count = sum(1 for e in st.session_state.health_history if "Online" in e["Status"])
        total_count = len(st.session_state.health_history)

        st.metric(
            "Uptime (recent checks)",
            f"{online_count}/{total_count} ({(online_count / total_count) * 100:.0f}%)"
        )

    else:
        st.info("No history yet — run a health check to start tracking.")


# ============================================================
# PAGE 4 — MODEL PERFORMANCE TRACKER
# ============================================================

elif page == "Model Performance Tracker":

    st.title("📊 Model Performance Tracker")
    st.caption("Compare accuracy, precision, recall, and F1 score across all trained models — no prediction input required.")

    st.divider()

    if st.button("🔄 Refresh Model Metrics", type="primary"):

        with st.spinner("Fetching model metrics..."):
            success, result, status_code, error, _ = call_api("GET", "/models")

        if success:
            st.session_state.models_info = result
        else:
            st.session_state.models_info = None
            st.error(
                f"Could not fetch model metrics: {error}\n\n"
                "Make sure your backend has a `GET /models` endpoint."
            )

    if st.session_state.models_info is not None:

        info = st.session_state.models_info

        st.success(f"Best Model: **{info.get('best_model')}**  |  Total Models: **{info.get('total_models')}**")

        models = info.get("models") or []

        if models:

            df = pd.DataFrame(models)

            df = df.rename(columns={
                "model": "Model",
                "is_best": "Best",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1 Score"
            })

            df = df.sort_values(by="F1 Score", ascending=False, na_position="last").reset_index(drop=True)

            display_df = df.copy()
            display_df["Best"] = display_df["Best"].apply(lambda v: "⭐" if v else "")

            for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                display_df[col] = display_df[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "N/A")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Metric Comparison")

            all_metrics_chart_df = df[
                ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
            ].dropna(how="all", subset=["Accuracy", "Precision", "Recall", "F1 Score"]).set_index("Model")

            if not all_metrics_chart_df.empty:
                st.caption("All Metrics Side by Side")
                st.bar_chart(all_metrics_chart_df)

            with st.expander("🔍 Focus on a single metric"):

                metric_choice = st.selectbox(
                    "Metric to chart",
                    options=["F1 Score", "Accuracy", "Precision", "Recall"],
                    index=0
                )

                single_chart_df = df[["Model", metric_choice]].dropna().set_index("Model")

                if not single_chart_df.empty:
                    st.bar_chart(single_chart_df)

        else:
            st.info("No model metrics available.")

    else:
        st.info("Click 'Refresh Model Metrics' to load performance data.")