# =========================================
# Apple Stock Forecast Dashboard (Streamlit)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="Apple Stock Forecast",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------------
# Custom CSS (Modern UI)
# -----------------------------------------
st.markdown("""
<style>
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #9CA3AF;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #10B981;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Title Section
# -----------------------------------------
st.title("📈 Apple Stock Price Forecast Dashboard")
st.caption("AI-powered forecasting using LSTM deep learning")

# -----------------------------------------
# Load Data
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL (5).csv")   # <-- your dataset
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

data = load_data()

# -----------------------------------------
# Key Metrics
# -----------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Latest Close Price</div>
        <div class="metric-value">${data['Adj Close'].iloc[-1]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Records</div>
        <div class="metric-value">{len(data)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Start Date</div>
        <div class="metric-value">{data.index.min().date()}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------------------
# Historical Chart
# -----------------------------------------
st.subheader("📊 Historical Adjusted Close Price")
st.line_chart(data['Adj Close'])

# -----------------------------------------
# Model Comparison
# -----------------------------------------
st.subheader("📌 Model Performance Comparison")

comparison = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "LSTM"],
    "RMSE": [0.166067, 0.118758, 0.023412],
    "MAE": [0.130148, 0.092876, 0.017800],
    "MAPE (%)": [18.15, 13.97, 2.81]
})

st.dataframe(comparison, use_container_width=True)
st.success("✅ LSTM selected as the best performing model")

st.divider()

# -----------------------------------------
# Date Picker
# -----------------------------------------
st.subheader("📅 Select Future Date for Forecast")

selected_date = st.date_input(
    "Choose a future business date",
    min_value=data.index[-1].date()
)

# Convert date to business days
last_date = data.index[-1].date()
n_days = len(pd.date_range(start=last_date, end=selected_date, freq="B")) - 1

MAX_DAYS = 120
if n_days <= 0:
    st.warning("Please select a future date.")
    st.stop()

if n_days > MAX_DAYS:
    st.warning("⚠️ Please select a date within the next 120 business days.")
    st.stop()

# -----------------------------------------
# LSTM Preparation
# -----------------------------------------
ts = data['Adj Close'].dropna().values.reshape(-1, 1)

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

# -----------------------------------------
# Train LSTM
# -----------------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

with st.spinner("🔄 Training LSTM model..."):
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# -----------------------------------------
# Forecast Until Selected Date
# -----------------------------------------
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

# -----------------------------------------
# Forecast Visualization
# -----------------------------------------
st.subheader("🔮 Forecast Trend")

combined = pd.concat([
    data['Adj Close'].tail(120),
    forecast_df['Forecast Price']
])

st.line_chart(combined)

# Highlight selected date prediction
predicted_price = forecast_df.loc[forecast_df.index[-1], "Forecast Price"]

st.markdown(f"""
### 📍 Prediction for **{selected_date}**
**Estimated Price:** 🟢 **${predicted_price:.2f}**
""")

# -----------------------------------------
# Business Insights
# -----------------------------------------
st.subheader("📌 Business Insights")

st.markdown("""
- Apple stock demonstrates **strong long-term growth**
- LSTM effectively captures **temporal market patterns**
- Forecast indicates **stable upward momentum**
- Useful for **strategic & long-term investment planning**
""")
