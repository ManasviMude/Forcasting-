# ===============================
# Apple Stock Forecast Dashboard
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Apple Stock Forecast",
    layout="wide"
)

st.title("📈 Apple Stock Price Forecast Dashboard")
st.markdown(
    "This dashboard analyzes **Apple Inc. stock prices** and forecasts "
    "future prices using **LSTM deep learning model**."
)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

data = load_data()

# -------------------------------
# Show Raw Data
# -------------------------------
st.subheader("📊 Historical Stock Data")
st.dataframe(data.tail())

# -------------------------------
# Plot Adjusted Close Price
# -------------------------------
st.subheader("📈 Adjusted Close Price Over Time")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(data.index, data['Adj Close'])
ax.set_xlabel("Date")
ax.set_ylabel("Adjusted Close Price")
ax.set_title("Apple Adjusted Close Price")
st.pyplot(fig)

# -------------------------------
# Model Comparison (From Task 2)
# -------------------------------
st.subheader("📌 Model Comparison")

comparison = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "LSTM"],
    "RMSE": [0.166067, 0.118758, 0.023412],
    "MAE": [0.130148, 0.092876, 0.017800],
    "MAPE (%)": [18.147808, 13.972719, 2.811019]
})

st.dataframe(comparison)
st.success("✅ LSTM selected as the best model based on lowest RMSE, MAE, and MAPE.")

# -------------------------------
# LSTM Forecast (Next 30 Days)
# -------------------------------
st.subheader("🔮 LSTM Forecast – Next 30 Days")

# Prepare time series
ts = data['Adj Close'].dropna().values.reshape(-1,1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ts)

# Create sequences
window = 60
X, y = [], []
for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train LSTM (lightweight for deployment)
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# Forecast next 30 days
last_sequence = scaled_data[-window:]
future_predictions = []

for _ in range(30):
    pred = model.predict(last_sequence.reshape(1, window, 1), verbose=0)
    future_predictions.append(pred[0,0])
    last_sequence = np.append(last_sequence[1:], pred)

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

future_dates = pd.date_range(
    start=data.index[-1],
    periods=31,
    freq='B'
)[1:]

# Plot forecast
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(data.index, data['Adj Close'], label="Historical")
ax.plot(future_dates, future_predictions, label="Forecast (30 Days)", linestyle="--")
ax.legend()
ax.set_title("30-Day Apple Stock Price Forecast")
st.pyplot(fig)

# -------------------------------
# Business Insights
# -------------------------------
st.subheader("📌 Business Insights")

st.markdown("""
- Apple stock shows a **strong long-term upward trend**
- LSTM outperforms traditional statistical models
- Forecast suggests **stable growth with moderate volatility**
- Useful for **long-term investment decision-making**
""")
