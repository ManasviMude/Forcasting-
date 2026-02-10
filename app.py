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
# Light macOS / iOS Style CSS
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f5f5f7;
}
.title-text {
    font-size: 36px;
    font-weight: 700;
    color: #1d1d1f;
}
.subtitle {
    color: #6e6e73;
    font-size: 16px;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}
.card-title {
    font-size: 14px;
    color: #6e6e73;
}
.card-value {
    font-size: 26px;
    font-weight: 600;
    color: #0071e3;
}
.section {
    margin-top: 35px;
}
.remark {
    background: #f0f8ff;
    padding: 16px;
    border-radius: 12px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("<div class='title-text'>Apple Stock Growth Forecast</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Historical analysis and future trend estimation</div>", unsafe_allow_html=True)

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
last_data_date = data.index[-1].date()

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
    <div class="card-title">Dataset Ends On</div>
    <div class="card-value">{last_data_date}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown("""
<div class="card">
    <div class="card-title">Selected Forecast Model</div>
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

st.markdown("""
<div class="remark">
<b>Why LSTM?</b><br>
Among all models, LSTM achieves the lowest RMSE, MAE, and MAPE values.
This indicates better accuracy and lower prediction error.
LSTM is well-suited for stock price forecasting because it captures long-term
temporal dependencies and non-linear patterns present in financial time series,
which traditional ARIMA-based models cannot model effectively.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Forecast Section
# --------------------------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("🔮 Forecast Future Stock Growth")

st.info(f"""
The historical dataset ends on **{last_data_date}**.  
Any date selected after this requires forecasting.
Predictions are limited to **120 business days** beyond the dataset
to maintain reasonable reliability.
""")

selected_date = st.date_input(
    "📅 Select a future business date",
    min_value=last_data_date
)

predict_btn = st.button("📈 Predict Stock Trend")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if predict_btn:

    n_days = len(pd.date_range(start=last_data_date, end=selected_date, freq="B")) - 1

    if n_days <= 0:
        st.warning("Please select a future date after the dataset end.")
        st.stop()

    if n_days > 120:
        st.warning("Please select a date within 120 business days after the dataset end.")
        st.stop()

    ts = data['Adj Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ts)

    window = 60
    X, y = [], []

    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_sequence = scaled_data[-window:]
    future_predictions = []

    for _ in range(n_days):
        pred = model.predict(last_sequence.reshape(1, window, 1), verbose=0)
        future_predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)

    future_prices = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    predicted_price = future_prices[-1][0]
    current_price = data['Adj Close'].iloc[-1]

    growth_pct = ((predicted_price - current_price) / current_price) * 100

    if growth_pct > 5:
        recommendation = "🟢 BUY"
    elif growth_pct < -5:
        recommendation = "🔴 SELL"
    else:
        recommendation = "🟡 HOLD"

    st.subheader("📉 Forecast Visualization")
    forecast_dates = pd.date_range(start=last_data_date, periods=n_days + 1, freq="B")[1:]
    forecast_df = pd.DataFrame(future_prices, index=forecast_dates, columns=["Forecast"])

    st.line_chart(pd.concat([data['Adj Close'].tail(120), forecast_df]))

    st.success(f"""
**Prediction Date:** {selected_date}  
**Predicted Price:** ${predicted_price:.2f}  
**Expected Growth:** {growth_pct:.2f}%  
**Recommendation:** {recommendation}
""")

# --------------------------------------------------
# Business Remarks
# --------------------------------------------------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📌 Remarks")

st.markdown("""
- Forecasting begins only after historical data ends  
- Growth percentage helps quantify future performance  
- Buy / Hold / Sell signal is rule-based and explainable  
- Suitable for academic analysis and decision support  
""")
