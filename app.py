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
# App Header
# --------------------------------------------------
st.title("🍎 Apple Stock Growth Forecast")
st.caption("Historical analysis and future trend estimation")

st.divider()

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

data = load_data()
last_data_date = data.index[-1].date()

# --------------------------------------------------
# Key Metrics (Clean & Native)
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Latest Price ($)", f"{data['Adj Close'].iloc[-1]:.2f}")
col2.metric("Total Trading Days", len(data))
col3.metric("Dataset Ends On", last_data_date)
col4.metric("Selected Model", "LSTM")

st.divider()

# --------------------------------------------------
# Historical Price Chart
# --------------------------------------------------
st.subheader("📈 Historical Adjusted Close Price")
st.line_chart(data["Adj Close"])

# --------------------------------------------------
# Model Comparison
# --------------------------------------------------
st.subheader("📊 Model Performance Comparison")

comparison = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "LSTM"],
    "RMSE": [0.1661, 0.1188, 0.0234],
    "MAE": [0.1301, 0.0929, 0.0178],
    "MAPE (%)": [18.15, 13.97, 2.81]
})

st.dataframe(comparison, use_container_width=True)

with st.expander("📌 Why LSTM was selected"):
    st.write(
        """
        LSTM has the lowest RMSE, MAE, and MAPE values among all models,
        indicating better predictive accuracy.
        
        Unlike ARIMA and SARIMA, LSTM can capture long-term dependencies
        and non-linear patterns in stock price data, making it more suitable
        for financial time series forecasting.
        """
    )

st.divider()

# --------------------------------------------------
# Forecast Section
# --------------------------------------------------
st.subheader("🔮 Forecast Future Stock Growth")

st.info(
    f"The historical dataset ends on **{last_data_date}**. "
    "Any date selected after this point requires forecasting."
)

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
        st.warning("Please select a valid future date.")
        st.stop()

    if n_days > 120:
        st.warning("Please select a date within 120 business days.")
        st.stop()

    # Prepare data
    ts = data["Adj Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ts)

    window = 60
    X, y = [], []

    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i - window:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    with st.spinner("Training LSTM model..."):
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Forecast
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
    current_price = data["Adj Close"].iloc[-1]
    growth_pct = ((predicted_price - current_price) / current_price) * 100

    # Recommendation
    if growth_pct > 5:
        recommendation = "🟢 BUY"
    elif growth_pct < -5:
        recommendation = "🔴 SELL"
    else:
        recommendation = "🟡 HOLD"

    # Forecast chart
    st.subheader("📉 Forecast Visualization")

    forecast_dates = pd.date_range(
        start=last_data_date, periods=n_days + 1, freq="B"
    )[1:]

    forecast_df = pd.DataFrame(
        future_prices, index=forecast_dates, columns=["Forecast"]
    )

    st.line_chart(pd.concat([data["Adj Close"].tail(120), forecast_df]))

    # Results
    st.success("### 📊 Forecast Summary")
    st.write(f"**Prediction Date:** {selected_date}")
    st.write(f"**Predicted Price:** ${predicted_price:.2f}")
    st.write(f"**Expected Growth:** {growth_pct:.2f}%")
    st.write(f"**Recommendation:** {recommendation}")

# --------------------------------------------------
# Remarks
# --------------------------------------------------
st.subheader("📌 Remarks")
st.write(
    """
    - Default Streamlit theme ensures clarity and consistency  
    - Forecasting starts after historical data ends  
    - Growth percentage quantifies future performance  
    - Buy/Hold/Sell recommendation is rule-based and interpretable  
    """
)
