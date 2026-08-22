import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
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
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

VISITOR_TYPE_OPTIONS = [
    "Returning_Visitor",
    "New_Visitor",
    "Other"
]

REQUEST_TIMEOUT = 10
MAX_HEALTH_HISTORY = 20

# Use environment variable if available.
# Otherwise use your deployed FastAPI backend.
DEFAULT_API_URL = os.getenv(
    "API_BASE_URL",
    "https://projects-prot.fastapicloud.dev"
)


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL

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
    Calls the FastAPI backend.

    Returns:
        success,
        payload,
        status_code,
        error_message,
        latency_ms
    """

    url = st.session_state.api_base_url.rstrip("/") + path

    start = time.perf_counter()

    try:

        if method.upper() == "GET":
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

        elif method.upper() == "POST":
            response = requests.post(
                url,
                json=json_body,
                timeout=REQUEST_TIMEOUT
            )

        else:
            return (
                False,
                None,
                None,
                f"Unsupported HTTP method: {method}",
                0
            )

    except requests.exceptions.ConnectionError:

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            1
        )

        return (
            False,
            None,
            None,
            (
                "Could not connect to the API. "
                "Make sure the FastAPI backend is running at "
                f"`{st.session_state.api_base_url}`."
            ),
            latency_ms
        )

    except requests.exceptions.Timeout:

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            1
        )

        return (
            False,
            None,
            None,
            "The request timed out.",
            latency_ms
        )

    except requests.exceptions.RequestException as e:

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            1
        )

        return (
            False,
            None,
            None,
            f"Request failed: {str(e)}",
            latency_ms
        )

    latency_ms = round(
        (time.perf_counter() - start) * 1000,
        1
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if response.status_code == 200:

        try:
            return (
                True,
                response.json(),
                response.status_code,
                None,
                latency_ms
            )

        except ValueError:

            return (
                False,
                None,
                response.status_code,
                "API returned an invalid JSON response.",
                latency_ms
            )

    # --------------------------------------------------------
    # ERROR RESPONSE
    # --------------------------------------------------------

    try:

        error_body = response.json()

        if isinstance(error_body, dict):
            detail = error_body.get(
                "detail",
                response.text
            )
        else:
            detail = response.text

    except ValueError:
        detail = response.text

    return (
        False,
        None,
        response.status_code,
        detail,
        latency_ms
    )


def format_pct(value):

    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "N/A"


def record_health_check():

    success, payload, status_code, error, latency_ms = call_api(
        "GET",
        "/health"
    )

    if success and isinstance(payload, dict):

        is_online = (
            payload.get("status") == "ok"
            and payload.get("model_loaded", False)
        )

        entry = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Status": "🟢 Online" if is_online else "🟠 API Online / Model Issue",
            "Latency (ms)": latency_ms,
            "Version": payload.get("version", "N/A")
        }

        st.session_state.health_status = payload

    else:

        entry = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Status": "🔴 Offline",
            "Latency (ms)": latency_ms,
            "Version": "N/A"
        }

        st.session_state.health_status = {
            "error": error
        }

    st.session_state.health_history.insert(
        0,
        entry
    )

    st.session_state.health_history = (
        st.session_state.health_history[:MAX_HEALTH_HISTORY]
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛒 Purchase & Recs")

    st.divider()

    st.subheader("⚙️ API Configuration")

    st.session_state.api_base_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="URL of your FastAPI backend."
    ).strip()

    if st.button(
        "🔌 Check API Health",
        use_container_width=True
    ):
        record_health_check()

    if st.session_state.health_status is not None:

        health = st.session_state.health_status

        if "error" in health:

            st.error(
                f"API unreachable: {health['error']}"
            )

        else:

            if (
                health.get("status") == "ok"
                and health.get("model_loaded")
            ):

                st.success(
                    f"API online — v{health.get('version', '?')}"
                )

            else:

                st.warning(
                    "API responded, but the model may not be loaded."
                )

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
            """
            This application connects to a FastAPI backend to:

            • Predict whether an online shopping session will result in a purchase.

            • Compare multiple machine-learning models.

            • Recommend similar products.

            • Monitor API health and response latency.

            • Track model performance metrics.
            """
        )


# ============================================================
# PAGE 1 — PURCHASE PREDICTION
# ============================================================

if page == "Purchase Prediction":

    st.title("🛍️ Purchase Prediction")

    st.caption(
        "Predict whether a shopping session will result in a purchase."
    )

    st.divider()

    with st.form("prediction_form"):

        # ----------------------------------------------------
        # SESSION BEHAVIOR
        # ----------------------------------------------------

        with st.expander(
            "📊 Session Behavior",
            expanded=True
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                administrative = st.number_input(
                    "Administrative Pages",
                    min_value=0,
                    step=1,
                    value=2
                )

                administrative_duration = st.number_input(
                    "Administrative Duration (sec)",
                    min_value=0.0,
                    step=1.0,
                    value=50.5
                )

            with col2:

                informational = st.number_input(
                    "Informational Pages",
                    min_value=0,
                    step=1,
                    value=1
                )

                informational_duration = st.number_input(
                    "Informational Duration (sec)",
                    min_value=0.0,
                    step=1.0,
                    value=20.0
                )

            with col3:

                product_related = st.number_input(
                    "Product-Related Pages",
                    min_value=0,
                    step=1,
                    value=30
                )

                product_related_duration = st.number_input(
                    "Product-Related Duration (sec)",
                    min_value=0.0,
                    step=1.0,
                    value=800.5
                )

        # ----------------------------------------------------
        # ENGAGEMENT
        # ----------------------------------------------------

        with st.expander(
            "📈 Engagement Metrics",
            expanded=True
        ):

            col1, col2 = st.columns(2)

            with col1:

                bounce_rates = st.slider(
                    "Bounce Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.02,
                    step=0.01
                )

                exit_rates = st.slider(
                    "Exit Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.04,
                    step=0.01
                )

            with col2:

                page_values = st.number_input(
                    "Page Values",
                    min_value=0.0,
                    step=0.5,
                    value=5.5,
                    help=(
                        "Average value of pages visited "
                        "before completing a transaction."
                    )
                )

                special_day = st.slider(
                    "Special Day Closeness",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help=(
                        "Closeness of the visit date to "
                        "a special day."
                    )
                )

        # ----------------------------------------------------
        # VISIT CONTEXT
        # ----------------------------------------------------

        with st.expander(
            "🌐 Visit Context",
            expanded=True
        ):

            col1, col2, col3 = st.columns(3)

            with col1:

                month = st.selectbox(
                    "Month",
                    options=MONTH_OPTIONS,
                    index=MONTH_OPTIONS.index("May")
                )

                visitor_type = st.selectbox(
                    "Visitor Type",
                    options=VISITOR_TYPE_OPTIONS,
                    index=VISITOR_TYPE_OPTIONS.index(
                        "Returning_Visitor"
                    )
                )

            with col2:

                operating_systems = st.number_input(
                    "Operating System (code)",
                    min_value=0,
                    step=1,
                    value=2
                )

                browser = st.number_input(
                    "Browser (code)",
                    min_value=0,
                    step=1,
                    value=1
                )

            with col3:

                region = st.number_input(
                    "Region (code)",
                    min_value=0,
                    step=1,
                    value=1
                )

                traffic_type = st.number_input(
                    "Traffic Type (code)",
                    min_value=0,
                    step=1,
                    value=2
                )

            weekend = st.checkbox(
                "Visit occurred on a weekend",
                value=False
            )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        with st.expander(
            "🤖 Model Selection",
            expanded=True
        ):

            model_name = st.selectbox(
                "Model",
                options=MODEL_OPTIONS,
                index=0,
                help=(
                    "Choose a specific model, or All Models "
                    "to compare every trained model."
                )
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔮 Predict",
            use_container_width=True,
            type="primary"
        )

    # ========================================================
    # SEND PREDICTION
    # ========================================================

    if submitted:

        payload = {
            "Administrative": int(administrative),
            "Administrative_Duration": float(
                administrative_duration
            ),
            "Informational": int(informational),
            "Informational_Duration": float(
                informational_duration
            ),
            "ProductRelated": int(product_related),
            "ProductRelated_Duration": float(
                product_related_duration
            ),
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

            (
                success,
                result,
                status_code,
                error,
                latency
            ) = call_api(
                "POST",
                "/predict",
                json_body=payload
            )

        if success:

            st.session_state.prediction_result = result

            st.success(
                f"Prediction completed in {latency} ms."
            )

        else:

            st.session_state.prediction_result = None

            if status_code == 422:

                st.error(
                    f"Validation error: {error}"
                )

            elif status_code == 500:

                st.error(
                    f"Server error: {error}"
                )

            else:

                st.error(
                    f"Prediction failed: {error}"
                )

    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    if st.session_state.prediction_result is not None:

        result = st.session_state.prediction_result

        st.divider()

        st.subheader("🎯 Prediction Result")

        will_purchase = result.get(
            "prediction"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            if will_purchase:

                st.success(
                    "### ✅ Will Purchase"
                )

            else:

                st.error(
                    "### ❌ Will Not Purchase"
                )

        with col2:

            st.metric(
                "Purchase Probability",
                format_pct(
                    result.get(
                        "purchase_probability"
                    )
                )
            )

        with col3:

            st.metric(
                "Confidence",
                format_pct(
                    result.get(
                        "confidence"
                    )
                )
            )

        st.caption(
            f"Model used: **{result.get('selected_model')}**"
        )

        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        model_results = (
            result.get("model_results") or []
        )

        if model_results:

            st.divider()

            st.subheader(
                f"🤖 Model Comparison ({len(model_results)} models)"
            )

            df = pd.DataFrame(
                model_results
            )

            df = df.rename(
                columns={
                    "model": "Model",
                    "prediction": "Prediction",
                    "purchase_probability": "Purchase Probability",
                    "confidence": "Confidence",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1 Score"
                }
            )

            if "F1 Score" in df.columns:

                df = df.sort_values(
                    by="F1 Score",
                    ascending=False,
                    na_position="last"
                ).reset_index(
                    drop=True
                )

            display_df = df.copy()

            for col in [
                "Purchase Probability",
                "Confidence",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]:

                if col in display_df.columns:

                    display_df[col] = display_df[
                        col
                    ].apply(
                        lambda v:
                        f"{v:.4f}"
                        if pd.notna(v)
                        else "N/A"
                    )

            if "Prediction" in display_df.columns:

                display_df["Prediction"] = (
                    display_df["Prediction"]
                    .apply(
                        lambda v:
                        "Purchase"
                        if v
                        else "No Purchase"
                    )
                )

            selected_model_name = result.get(
                "selected_model"
            )

            def highlight_selected(row):

                if row["Model"] == selected_model_name:

                    return [
                        "background-color: #e6f4ea"
                    ] * len(row)

                return [""] * len(row)

            st.dataframe(
                display_df.style.apply(
                    highlight_selected,
                    axis=1
                ),
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # PLOTLY MODEL METRICS
            # =================================================

            if len(df) > 1:

                metric_cols = [
                    c
                    for c in [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score"
                    ]
                    if c in df.columns
                ]

                if metric_cols:

                    chart_df = (
                        df[
                            ["Model"] + metric_cols
                        ]
                        .dropna(
                            how="all",
                            subset=metric_cols
                        )
                    )

                    if not chart_df.empty:

                        st.subheader(
                            "📊 Model Performance"
                        )

                        long_df = chart_df.melt(
                            id_vars="Model",
                            value_vars=metric_cols,
                            var_name="Metric",
                            value_name="Score"
                        )

                        fig = px.bar(
                            long_df,
                            x="Model",
                            y="Score",
                            color="Metric",
                            barmode="group",
                            title="Accuracy, Precision, Recall & F1 Score"
                        )

                        fig.update_layout(
                            yaxis=dict(
                                range=[0, 1]
                            ),
                            height=500,
                            legend_title="Metric"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        recommendations = (
            result.get("recommendations") or []
        )

        if recommendations:

            st.divider()

            st.subheader(
                "🎁 Related Recommendations"
            )

            st.dataframe(
                pd.DataFrame(
                    recommendations
                ),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# PAGE 2 — PRODUCT RECOMMENDATIONS
# ============================================================

elif page == "Product Recommendations":

    st.title(
        "🎁 Product Recommendations"
    )

    st.caption(
        "Find products similar to a given stock code, "
        "based on customer purchase patterns."
    )

    st.divider()

    with st.form(
        "recommendation_form"
    ):

        col1, col2 = st.columns(
            [3, 1]
        )

        with col1:

            stock_code = st.text_input(
                "Stock Code",
                placeholder="e.g. 85123A",
                help="The product's stock/SKU code"
            )

        with col2:

            n = st.number_input(
                "Number of Recommendations",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )

        submitted = st.form_submit_button(
            "🔍 Get Recommendations",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        if not stock_code.strip():

            st.warning(
                "Please enter a stock code."
            )

        else:

            payload = {
                "stock_code": stock_code.strip(),
                "n": int(n)
            }

            with st.spinner(
                "Finding similar products..."
            ):

                (
                    success,
                    result,
                    status_code,
                    error,
                    latency
                ) = call_api(
                    "POST",
                    "/recommend",
                    json_body=payload
                )

            if success:

                st.session_state.recommendation_result = result

                st.success(
                    f"Recommendations loaded in {latency} ms."
                )

            else:

                st.session_state.recommendation_result = None

                if status_code == 404:

                    st.warning(
                        f"⚠️ {error}"
                    )

                elif status_code == 422:

                    st.error(
                        f"Validation error: {error}"
                    )

                else:

                    st.error(
                        f"Recommendation failed: {error}"
                    )

    # ========================================================
    # RESULTS
    # ========================================================

    if st.session_state.recommendation_result is not None:

        result = (
            st.session_state.recommendation_result
        )

        recommendations = (
            result.get(
                "recommendations"
            ) or []
        )

        st.divider()

        st.subheader(
            f"Results for '{result.get('stock_code')}'"
        )

        st.caption(
            f"{result.get('number_of_recommendations', len(recommendations))} recommendations found"
        )

        if recommendations:

            df = pd.DataFrame(
                recommendations
            )

            if "Similarity" in df.columns:

                df = df.sort_values(
                    by="Similarity",
                    ascending=False
                ).reset_index(
                    drop=True
                )

            display_df = df.copy()

            if "Similarity" in display_df.columns:

                display_df["Similarity"] = (
                    display_df["Similarity"]
                    .apply(
                        lambda v:
                        f"{v * 100:.1f}%"
                        if pd.notna(v)
                        else "N/A"
                    )
                )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # PLOTLY RECOMMENDATION CHART
            # =================================================

            if "Similarity" in df.columns:

                chart_label_col = (
                    "Description"
                    if "Description" in df.columns
                    else "StockCode"
                )

                chart_df = (
                    df[
                        [
                            chart_label_col,
                            "Similarity"
                        ]
                    ]
                    .dropna()
                    .copy()
                )

                if not chart_df.empty:

                    chart_df = chart_df.sort_values(
                        "Similarity",
                        ascending=True
                    )

                    st.subheader(
                        "📊 Product Similarity"
                    )

                    fig = px.bar(
                        chart_df,
                        x="Similarity",
                        y=chart_label_col,
                        orientation="h",
                        title="Recommended Products by Similarity",
                        labels={
                            "Similarity": "Similarity Score",
                            chart_label_col: "Product"
                        }
                    )

                    fig.update_layout(
                        xaxis=dict(
                            range=[0, 1]
                        ),
                        height=max(
                            400,
                            len(chart_df) * 45
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        else:

            st.info(
                "No recommendations returned."
            )


# ============================================================
# PAGE 3 — MODEL HEALTH TRACKER
# ============================================================

elif page == "Model Health Tracker":

    st.title(
        "🩺 Model Health Tracker"
    )

    st.caption(
        "Live status checks of the backend API "
        "and model availability over time."
    )

    st.divider()

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:

        if st.button(
            "🔄 Run Health Check Now",
            use_container_width=True,
            type="primary"
        ):

            record_health_check()

    with col2:

        if (
            st.session_state.health_status
            and "error"
            not in st.session_state.health_status
        ):

            h = st.session_state.health_status

            if (
                h.get("status") == "ok"
                and h.get("model_loaded")
            ):

                st.success(
                    f"Status: **{h.get('status')}**  |  "
                    f"Model Loaded: **{h.get('model_loaded')}**  |  "
                    f"Version: **{h.get('version')}**"
                )

            else:

                st.warning(
                    f"API responded, but model status needs attention. "
                    f"Version: **{h.get('version')}**"
                )

        elif st.session_state.health_status:

            st.error(
                "Last check failed: "
                f"{st.session_state.health_status.get('error')}"
            )

        else:

            st.info(
                "No health checks recorded yet. "
                "Click the button to run one."
            )

    st.divider()

    if st.session_state.health_history:

        st.subheader(
            "📋 Check History"
        )

        history_df = pd.DataFrame(
            st.session_state.health_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # PLOTLY LATENCY CHART
        # ====================================================

        latency_df = (
            history_df[
                ["Time", "Latency (ms)"]
            ]
            .copy()
        )

        if not latency_df.empty:

            latency_df = latency_df.iloc[::-1]

            st.subheader(
                "⚡ API Response Latency"
            )

            fig = px.line(
                latency_df,
                x="Time",
                y="Latency (ms)",
                markers=True,
                title="Response Latency Over Recent Checks"
            )

            fig.update_layout(
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        online_count = sum(
            1
            for e in st.session_state.health_history
            if "Online" in e["Status"]
        )

        total_count = len(
            st.session_state.health_history
        )

        uptime = (
            online_count / total_count
        ) * 100

        st.metric(
            "Uptime (Recent Checks)",
            f"{online_count}/{total_count} ({uptime:.0f}%)"
        )

    else:

        st.info(
            "No history yet — run a health check "
            "to start tracking."
        )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE TRACKER
# ============================================================

elif page == "Model Performance Tracker":

    st.title(
        "📊 Model Performance Tracker"
    )

    st.caption(
        "Compare accuracy, precision, recall, and F1 score "
        "across all trained models."
    )

    st.divider()

    if st.button(
        "🔄 Refresh Model Metrics",
        type="primary"
    ):

        with st.spinner(
            "Fetching model metrics..."
        ):

            (
                success,
                result,
                status_code,
                error,
                latency
            ) = call_api(
                "GET",
                "/models"
            )

        if success:

            st.session_state.models_info = result

            st.success(
                f"Metrics loaded in {latency} ms."
            )

        else:

            st.session_state.models_info = None

            st.error(
                f"Could not fetch model metrics: {error}\n\n"
                "Make sure your backend has a GET /models endpoint."
            )

    if st.session_state.models_info is not None:

        info = st.session_state.models_info

        st.success(
            f"Best Model: **{info.get('best_model')}**  |  "
            f"Total Models: **{info.get('total_models')}**"
        )

        models = (
            info.get("models") or []
        )

        if models:

            df = pd.DataFrame(
                models
            )

            df = df.rename(
                columns={
                    "model": "Model",
                    "is_best": "Best",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1 Score"
                }
            )

            if "F1 Score" in df.columns:

                df = df.sort_values(
                    by="F1 Score",
                    ascending=False,
                    na_position="last"
                ).reset_index(
                    drop=True
                )

            display_df = df.copy()

            if "Best" in display_df.columns:

                display_df["Best"] = (
                    display_df["Best"]
                    .apply(
                        lambda v:
                        "⭐"
                        if v
                        else ""
                    )
                )

            for col in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]:

                if col in display_df.columns:

                    display_df[col] = (
                        display_df[col]
                        .apply(
                            lambda v:
                            f"{v:.4f}"
                            if pd.notna(v)
                            else "N/A"
                        )
                    )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "📊 Metric Comparison"
            )

            metric_columns = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]

            available_metrics = [
                col
                for col in metric_columns
                if col in df.columns
            ]

            if available_metrics:

                chart_df = (
                    df[
                        ["Model"] + available_metrics
                    ]
                    .dropna(
                        how="all",
                        subset=available_metrics
                    )
                )

                if not chart_df.empty:

                    long_df = chart_df.melt(
                        id_vars="Model",
                        value_vars=available_metrics,
                        var_name="Metric",
                        value_name="Score"
                    )

                    fig = px.bar(
                        long_df,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Model Performance Comparison"
                    )

                    fig.update_layout(
                        yaxis=dict(
                            range=[0, 1]
                        ),
                        height=500
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                # =================================================
                # SINGLE METRIC
                # =================================================

                with st.expander(
                    "🔍 Focus on a Single Metric"
                ):

                    metric_choice = st.selectbox(
                        "Metric to chart",
                        options=available_metrics,
                        index=(
                            available_metrics.index(
                                "F1 Score"
                            )
                            if "F1 Score"
                            in available_metrics
                            else 0
                        )
                    )

                    single_chart_df = (
                        df[
                            [
                                "Model",
                                metric_choice
                            ]
                        ]
                        .dropna()
                    )

                    if not single_chart_df.empty:

                        fig = px.bar(
                            single_chart_df,
                            x="Model",
                            y=metric_choice,
                            title=f"{metric_choice} by Model",
                            text=metric_choice
                        )

                        fig.update_layout(
                            yaxis=dict(
                                range=[0, 1]
                            ),
                            height=450
                        )

                        fig.update_traces(
                            texttemplate="%{text:.3f}",
                            textposition="outside"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

        else:

            st.info(
                "No model metrics available."
            )

    else:

        st.info(
            "Click 'Refresh Model Metrics' "
            "to load performance data."
        )
