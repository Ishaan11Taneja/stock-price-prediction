import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from src.data_loader import load_tata_global_data
from src.feature_engineering import add_technical_indicators
from src.evaluation import regression_metrics
from src.models import (train_predict_rf, train_predict_xgb, train_predict_prophet)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# --- Page Config ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- Caching Functions ---
@st.cache_data
def load_data():
    df = load_tata_global_data()
    df = add_technical_indicators(df)
    df = df.dropna()
    X = df.drop(columns=['Date', 'Close'])
    X = X.ffill().bfill()
    y = df['Close']
    return df, X, y

@st.cache_data
def get_model_predictions(X, y):
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # --- Linear Regression ---
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # --- RF and XGB (Differenced) ---
    y_train_diff = y_train.diff().dropna()
    X_train_diff = X_train.loc[y_train_diff.index]
    _, rf_model = train_predict_rf(X_train_diff, y_train_diff, X_test)
    _, xgb_model = train_predict_xgb(X_train_diff, y_train_diff, X_test)

    # Return models for forecasting
    return lr_model, rf_model, xgb_model, X_train_diff, y_train, y_test, X_test

@st.cache_data
def get_7_day_forecasts(_lr_model, _rf_model, _xgb_model, _X_test, _y_test):
    forecasts = {}
    last_features = _X_test.iloc[[-1]].copy()
    last_close = _y_test.iloc[-1]

    # LR
    preds_lr = []
    current_features_lr = last_features.copy()
    for _ in range(7):
        pred = _lr_model.predict(current_features_lr)[0]
        preds_lr.append(pred)
    forecasts['Linear Regression'] = preds_lr

    # RF
    preds_rf = []
    current_features_rf = last_features.copy()
    iter_close_rf = last_close
    for _ in range(7):
        pred_diff = _rf_model.predict(current_features_rf)[0]
        new_close = iter_close_rf + pred_diff
        preds_rf.append(new_close)
        iter_close_rf = new_close
    forecasts['Random Forest'] = preds_rf

    # XGB
    preds_xgb = []
    current_features_xgb = last_features.copy()
    iter_close_xgb = last_close
    for _ in range(7):
        pred_diff = _xgb_model.predict(current_features_xgb)[0]
        new_close = iter_close_xgb + pred_diff
        preds_xgb.append(new_close)
        iter_close_xgb = new_close
    forecasts['XGBoost'] = preds_xgb
    
    return forecasts

@st.cache_data
def get_prophet_forecast(_df):
    forecast, _ = train_predict_prophet(_df, 7)
    return forecast

@st.cache_data
def get_shap_values(_xgb_model, _X_train_diff):
    explainer = shap.TreeExplainer(_xgb_model)
    shap_values = explainer.shap_values(_X_train_diff)
    return shap_values

# --- Main App ---
st.title("📈 Stock Price Prediction Dashboard")
st.write("""
This application demonstrates various machine learning models to predict stock prices 
and provides a 7-day forecast.
""")

# Load and process data
df, X, y = load_data()
lr_model, rf_model, xgb_model, X_train_diff, y_train, y_test, X_test = get_model_predictions(X, y)

# --- Display Data ---
st.header("Historical Stock Data (TATA GLOBAL)")
st.dataframe(df.tail())

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Close'], label='Close Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.set_title("TATA GLOBAL Historical Close Price")
ax.grid(True)
st.pyplot(fig)

# --- 7-Day Forecast ---
st.header("🔮 7-Day Price Forecast")
forecasts = get_7_day_forecasts(lr_model, rf_model, xgb_model, X_test, y_test)
prophet_forecast = get_prophet_forecast(df)

forecast_df = pd.DataFrame(forecasts)
forecast_df['Date'] = prophet_forecast['ds'].values
forecast_df['Prophet'] = prophet_forecast['yhat'].values
forecast_df = forecast_df.set_index('Date')

st.dataframe(forecast_df)

fig2, ax2 = plt.subplots(figsize=(12, 6))
for model_name in forecast_df.columns:
    ax2.plot(forecast_df.index, forecast_df[model_name], marker='o', linestyle='--', label=model_name)
ax2.set_xlabel("Date")
ax2.set_ylabel("Predicted Price (INR)")
ax2.set_title("All Models: 7-Day Forecast")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# --- SHAP Interpretation ---
st.header("🧠 Model Interpretation (SHAP for XGBoost)")
st.write("""
SHAP (SHapley Additive exPlanations) values show the impact of each feature on the model's output. 
This plot illustrates the average importance of each feature for the XGBoost model.
""")
shap_values = get_shap_values(xgb_model, X_train_diff)

fig3, ax3 = plt.subplots(figsize=(10, 8))
plt.title('SHAP Feature Importance for XGBoost')
shap.summary_plot(shap_values, X_train_diff, plot_type="bar", show=False)
st.pyplot(fig3) 