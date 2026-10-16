
import time

import pandas as pd
import requests
import streamlit as st

from streamlit_autorefresh import st_autorefresh

# --------------------------------------------------
# Configuration
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 5

st.set_page_config(
    page_title="ML Model Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Professional light theme
# --------------------------------------------------

st.markdown(
    """
<style>
.stApp {
    background-color: #F3F6FC;
    color: #172033;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] * {
    color: #172033 !important;
}

.hero h1 {
    color: #12346A !important;
}

.hero p {
    color: #334B70 !important;
}

.section-title {
    color: #172033 !important;
}

.card-label {
    color: #475569 !important;
}

.card-value {
    color: #102A56 !important;
}

.card-note {
    color: #526581 !important;
}

.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #172033 !important;
}

/* Separate cards for health and metrics */
.info-card {
    background-color: #FFFFFF;
    border: 1px solid #D5DEEC;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 4px 12px rgba(30, 50, 80, 0.07);
    margin-bottom: 12px;
}

.card-label {
    color: #52627A !important;
    font-size: 14px;
    margin-bottom: 12px;
}

.card-value {
    color: #17365D !important;
    font-size: 25px;
    font-weight: 700;
    overflow-wrap: anywhere;
}

.card-note {
    color: #64748B !important;
    font-size: 12px;
    margin-top: 8px;
}

/* Keep spacing between metric cards */
[data-testid="column"] {
    padding: 4px;
}

/* White prediction form */
div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #D5DEEC;
    border-radius: 14px;
    padding: 20px;
}

/* Readable input fields */
.stNumberInput input,
.stTextInput input {
    background-color: #FFFFFF !important;
    color: #172033 !important;
    border: 1px solid #AAB8CC !important;
}

/* Improve card spacing */
[data-testid="column"] {
    padding: 4px;
}


/* Make prediction results clearly readable */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D5DEEC;
    border-radius: 14px;
    padding: 18px !important;
    box-shadow: 0 4px 12px rgba(30, 50, 80, 0.06);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #12346A !important;
    font-weight: 750 !important;
}

[data-testid="stMetricDelta"] {
    color: #166534 !important;
}

.result-card {
    background-color: #FFFFFF !important;
    border: 1px solid #C5D7F5;
    border-radius: 16px;
    padding: 24px;
    margin-top: 18px;
}

.result-label {
    color: #334155 !important;
    font-weight: 600;
}


</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📊 ML Monitor")
    st.caption("Model deployment dashboard")

    st.divider()

    st.markdown("**Project**")
    st.write("Iris Classification")

    st.markdown("**Backend**")
    st.write("FastAPI + Docker")

    st.markdown("**Tracking**")
    st.write("Request counts, errors and latency")

    st.divider()

    st.caption("Dashboard v1.0")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>ML Deployment &amp; Monitoring</h1>
        <p>
            Monitor model health, API performance and Iris predictions
            from one dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Refresh controls
# --------------------------------------------------
control_col, auto_col, time_col = st.columns([1, 1, 2])

with control_col:
    if st.button("↻ Refresh dashboard", use_container_width=True):
        st.rerun()

with auto_col:
    auto_refresh = st.toggle("Auto-refresh (15s)", value=False)

with time_col:
    st.caption(
        f"Last page refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

if auto_refresh:
    st_autorefresh(interval=15_000, key="ml_metrics_refresh")


# --------------------------------------------------
# Fetch API health and metrics
# --------------------------------------------------
health = None
metrics = None
api_error = None

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=REQUEST_TIMEOUT,
    )
    health_response.raise_for_status()
    health = health_response.json()

    metrics_response = requests.get(
        f"{API_URL}/metrics",
        timeout=REQUEST_TIMEOUT,
    )
    metrics_response.raise_for_status()
    metrics = metrics_response.json()

except requests.RequestException as exc:
    api_error = str(exc)

except ValueError as exc:
    api_error = f"Invalid JSON returned by the API: {exc}"


# --------------------------------------------------
# Health section
# --------------------------------------------------
st.markdown(
    '<div class="section-title">System Health</div>',
    unsafe_allow_html=True,
)

if api_error:
    st.error(
        "Cannot connect to the FastAPI service. "
        "Check that your ml-platform-api Docker container is running."
    )
    with st.expander("Connection details"):
        st.code(api_error)

else:
    is_healthy = (
        health.get("status") == "healthy"
        and health.get("model_loaded") is True
    )

    health_text = "Healthy" if is_healthy else "Unhealthy"
    health_class = "status-good" if is_healthy else "status-bad"
    model_type = health.get("model_type") or "Unknown"

    health_col, model_col = st.columns(2)

    with health_col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">API Status</div>
                <div class="card-value {health_class}">
                    {health_text}
                </div>
                <div class="card-note">
                    FastAPI health endpoint
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with model_col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">Loaded Model</div>
                <div class="card-value">{model_type}</div>
                <div class="card-note">
                    Model reported by the API
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# Metrics section
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Prediction Metrics</div>',
    unsafe_allow_html=True,
)

if metrics is not None:
    requests_total = int(
        metrics.get("prediction_requests_total", 0)
    )
    errors_total = int(
        metrics.get("prediction_errors_total", 0)
    )
    latency = float(
        metrics.get("average_prediction_latency_ms", 0.0)
    )

    error_rate = (
        errors_total / requests_total * 100
        if requests_total > 0
        else 0.0
    )

    cards = st.columns(4)

    card_data = [
        ("Total Requests", f"{requests_total:,}", "All recorded prediction requests"),
        ("Prediction Errors", f"{errors_total:,}", "Recorded failed requests"),
        ("Error Rate", f"{error_rate:.2f}%", "Errors divided by requests"),
        ("Average Latency", f"{latency:.2f} ms", "Reported prediction latency"),
    ]

    for column, (label, value, note) in zip(cards, card_data):
        with column:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-label">{label}</div>
                    <div class="card-value">{value}</div>
                    <div class="card-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    if requests_total > 0:
        st.progress(
            min(errors_total / requests_total, 1.0),
            text="Share of requests recorded as errors",
        )
    else:
        st.progress(0.0, text="No prediction requests recorded yet")

    st.caption(
        "Counters belong to the running API instance and may reset "
        "when that API process restarts."
    )


# --------------------------------------------------
# Iris prediction form
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-title">🌸 Iris Species Prediction</div>',
    unsafe_allow_html=True,
)

st.write(
    "Enter the flower measurements below and submit them to your "
    "deployed Random Forest model."
)

with st.form("iris_prediction_form"):
    left_col, right_col = st.columns(2)

    with left_col:
        sepal_length = st.number_input(
            "Sepal length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.1,
            step=0.1,
            format="%.2f",
        )

        sepal_width = st.number_input(
            "Sepal width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
            format="%.2f",
        )

    with right_col:
        petal_length = st.number_input(
            "Petal length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.4,
            step=0.1,
            format="%.2f",
        )

        petal_width = st.number_input(
            "Petal width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
            format="%.2f",
        )

    submitted = st.form_submit_button(
        "Predict Iris Species",
        type="primary",
        use_container_width=True,
    )


# Keep the latest result visible across dashboard reruns.
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_prediction_inputs" not in st.session_state:
    st.session_state.last_prediction_inputs = None

if submitted:
    payload = {
        "features": {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width,
        }
    }

    try:
        with st.spinner("Getting prediction from the API..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

        st.session_state.last_prediction = result
        st.session_state.last_prediction_inputs = payload["features"]

    except requests.HTTPError as exc:
        st.session_state.last_prediction = None

        details = ""
        if exc.response is not None:
            details = exc.response.text

        st.error("The API rejected the prediction request.")
        if details:
            st.code(details)

    except requests.RequestException as exc:
        st.session_state.last_prediction = None
        st.error("Prediction failed because the API could not be reached.")
        st.code(str(exc))

    except ValueError as exc:
        st.session_state.last_prediction = None
        st.error("The API returned an invalid prediction response.")
        st.code(str(exc))


# --------------------------------------------------
# Display the latest prediction
# --------------------------------------------------
result = st.session_state.last_prediction

if result:
    prediction = result.get("prediction", "Unknown")
    class_id = result.get("class_id", "N/A")
    confidence = result.get("confidence")

    st.markdown(
        """
        <div class="result-card">
            <div class="result-label">Latest prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_col, class_col, confidence_col = st.columns(3)

    with result_col:
        st.metric("Predicted Species", str(prediction).title())

    with class_col:
        st.metric("Class ID", str(class_id))

    with confidence_col:
        if isinstance(confidence, (int, float)):
            confidence_display = f"{confidence * 100:.1f}%"
        else:
            confidence_display = "Not provided"

        st.metric("Model Confidence", confidence_display)

    with st.expander("View submitted measurements"):
        st.json(st.session_state.last_prediction_inputs)

    with st.expander("View raw API response"):
        st.json(result)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "ML Deployment & Monitoring Platform • "
    "Streamlit frontend • FastAPI backend • Random Forest model"
)


# Prediction History
st.subheader("📊 Prediction History")

try:
    response = requests.get(
        "http://127.0.0.1:8000/history",
        timeout=5
    )

    if response.status_code == 200:
        history = response.json()

        if history:
            df = pd.DataFrame(history)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            if "prediction" in df.columns:
                st.bar_chart(df["prediction"].value_counts())
        else:
            st.info("No prediction history available yet.")
    else:
        st.warning("Could not load prediction history.")

except requests.RequestException:
    st.warning("API is unavailable. Please start the API.")


# ==========================================
# MODEL MONITORING
# ==========================================
st.divider()
st.header("🔬 Model Performance Monitoring")

try:
    response = requests.get(
        "http://127.0.0.1:8000/monitoring",
        timeout=5
    )
    response.raise_for_status()
    monitoring = response.json()

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Recorded Predictions",
            monitoring["total_predictions"]
        )

    with metric2:
        st.metric(
            "Average Confidence",
            f'{monitoring["average_confidence"]:.1%}'
        )

    with metric3:
        st.metric(
            "Species Classes",
            len(monitoring["distribution"])
        )

    st.subheader("Prediction Distribution")

    distribution = monitoring["distribution"]

    if distribution:
        distribution_df = pd.DataFrame(
            list(distribution.items()),
            columns=["Species", "Predictions"]
        ).set_index("Species")

        st.bar_chart(distribution_df)
    else:
        st.info("Make predictions to start monitoring.")

    st.subheader("Confidence Trend")

    trend = monitoring["confidence_trend"]

    if trend:
        trend_df = pd.DataFrame(trend)
        trend_df["timestamp"] = pd.to_datetime(
            trend_df["timestamp"], errors="coerce"
        )
        trend_df = trend_df.dropna(subset=["timestamp"])
        trend_df = trend_df.set_index("timestamp")

        st.line_chart(trend_df["confidence"])
    else:
        st.info("Confidence trends will appear after predictions.")

    st.subheader("Prediction Drift Alert")

    if monitoring["drift_detected"]:
        st.warning("⚠️ " + monitoring["alert"])
    elif monitoring["total_predictions"] >= 40:
        st.success("✅ " + monitoring["alert"])
    else:
        st.info("ℹ️ " + monitoring["alert"])

except requests.RequestException as exc:
    st.error(f"Monitoring API unavailable: {exc}")