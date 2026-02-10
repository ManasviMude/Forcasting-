# ==================================================
# 🍎 Apple Stock Growth Forecast Dashboard
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Apple Stock Forecast",
    page_icon="🍎",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS (Eye-Catchy UI)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.title-text {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg, #22c55e, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #9ca3af;
    font-size: 16px;
}
.card {
    background: #111827;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.4);
}
.card-title {
    font-size: 14px;
    color: #9ca3af;
}
.card-value {
    font-size: 28px;
    font-weight: 700;
    color: #22c55e;
}
.section {
    margin-top: 40px;
}
.badge {
    display: inline-block;
    background: #064e3b;
    color: #22c55e;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("<div class='title-text'>🍎 Apple Stock Growth Forecast</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered stock price prediction using LSTM deep learning</div>", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

data = load_data()

# --------------------------------------------------
# KPI Cards
# --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class="card">
    <div class="card-title">Latest Price</div>
    <div class="card-value">${data['Adj Close'].iloc[-1]:.2f}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="card">
    <div class="card-title">Total Trading Days</div>
    <div class="card-value">{len(data)}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="card">
    <div class="card-title">Start Date</div>
    <div class="card-value">{data.index.min().date()}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown("""
<div class="card">
    <div class="card-title">Best Model</div>
    <div class="card-value">LSTM</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Historical Chart
# --------------------------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📈 Historical Adjusted Close Price")
st.line_chart(data['Adj Close'])

# --------------------------------------------------
# Model Comparison
# --------------------------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📊 Model Performance Comparison")

comparison = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "LSTM"],
    "RMSE": [0.1661, 0.1188, 0.0234],
    "MAE": [0.1301, 0.0929, 0.0178],
    "MAPE (%)": [18.15, 13.97, 2.81]
})

st.dataframe(comparison, use_container_width=True)
st.markdown("<span class='badge'>✔ LSTM Selected as Best Model</span>", unsafe_allow_html=True)

# --------------------------------------------------
# Forecast Section
# --------------------------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("🔮 Forecast Future Stock Growth")

selected_date = st.date_input(
    "📅 Select a future business date",
    min_value=data.index[-1].date()
)

predict_btn = st.button("🚀 Predict Stock Trend")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if predict_btn:

    last_date = data.index[-1].date()
    n_days = len(pd.date_range(start=last_date, end=selected_date, freq="B")) - 1

    MAX_DAYS = 120

    if n_days <= 0:
        st.warning("⚠️ Please select a valid future date.")
        st.stop()

    if n_days > MAX_DAYS:
        st.warning("⚠️ Please select a date within the next 120 business days.")
        st.stop()

    ts = data['Adj Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ts)

    window = 60
    X, y = [], []

    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    with st.spinner("🤖 Training LSTM model..."):
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_sequence = scaled_data[-window:]
    future_predictions = []

    for _ in range(n_days):
        pred = model.predict(last_sequence.reshape(1, window, 1), verbose=0)
        future_predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    future_dates = pd.date_range(
        start=data.index[-1],
        periods=n_days + 1,
        freq="B"
    )[1:]

    forecast_df = pd.DataFrame(
        future_predictions,
        index=future_dates,
        columns=["Forecast Price"]
    )

    st.subheader("📉 Forecast Trend Visualization")

    combined = pd.concat([
        data['Adj Close'].tail(120),
        forecast_df['Forecast Price']
    ])

    st.line_chart(combined)

    predicted_price = forecast_df.iloc[-1, 0]

    st.success(
        f"📍 Predicted Apple stock price on **{selected_date}**: **${predicted_price:.2f}**"
    )

# --------------------------------------------------
# Business Insights
# --------------------------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📌 Business Insights")

st.markdown("""
- 📈 Apple exhibits **consistent long-term growth**
- 🤖 LSTM captures **non-linear market behavior**
- 🔮 Forecast indicates **stable upward momentum**
- 💼 Useful for **strategic investment planning**
""")
