import os
from src.data_loader import load_tata_global_data
from src.feature_engineering import add_technical_indicators
from src.evaluation import regression_metrics
from src.utils import plot_actual_vs_predicted
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from src.models import (train_predict_rf, train_predict_xgb, train_predict_lstm, 
                          recursive_multi_step_lstm, train_predict_seq2seq, train_predict_prophet)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load data
df = load_tata_global_data()
df = add_technical_indicators(df)

# Drop rows with NaN values (from indicators)
df = df.dropna().reset_index(drop=True)

# Features and target
features = ['Open', 'High', 'Low', 'Total Trade Quantity', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'MACD_Signal']
target = 'Close'

X = df[features]
y = df[target]

# Forward-fill and back-fill any remaining NaNs to ensure data is clean
X = X.ffill().bfill()

# Train-test split (last 20% as test set for time series)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df['Date'].iloc[split_idx:]

# Baseline Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)
metrics_lr = regression_metrics(y_test, y_pred_lr)
print('Linear Regression Performance:')
for k, v in metrics_lr.items():
    print(f'{k}: {v:.4f}')
plot_actual_vs_predicted(dates_test, y_test, y_pred_lr, title='Linear Regression: Actual vs Predicted Close Prices')

# --- Random Forest (Differenced Model) ---
print('\nRandom Forest Performance (on price differences):')
# Create differenced target
y_train_diff = y_train.diff().dropna()
X_train_diff = X_train.loc[y_train_diff.index]
# Train on differences
y_pred_rf_diff, rf_model = train_predict_rf(X_train_diff, y_train_diff, X_test)
# Reconstruct price prediction
prev_close_rf = df['Close'].loc[y_test.index - 1].values
y_pred_rf = prev_close_rf + y_pred_rf_diff
# Evaluate on reconstructed price
metrics_rf = regression_metrics(y_test, y_pred_rf)
for k, v in metrics_rf.items():
    print(f'{k}: {v:.4f}')
plot_actual_vs_predicted(dates_test, y_test, y_pred_rf, title='Random Forest: Actual vs Predicted Close Prices')

# --- XGBoost (Differenced Model) ---
print('\nXGBoost Performance (on price differences):')
# Create differenced target (same as for RF)
y_train_diff = y_train.diff().dropna()
X_train_diff = X_train.loc[y_train_diff.index]
# Train on differences
y_pred_xgb_diff, xgb_model = train_predict_xgb(X_train_diff, y_train_diff, X_test)
# Reconstruct price prediction
prev_close_xgb = df['Close'].loc[y_test.index - 1].values
y_pred_xgb = prev_close_xgb + y_pred_xgb_diff
# Evaluate on reconstructed price
metrics_xgb = regression_metrics(y_test, y_pred_xgb)
for k, v in metrics_xgb.items():
    print(f'{k}: {v:.4f}')
plot_actual_vs_predicted(dates_test, y_test, y_pred_xgb, title='XGBoost: Actual vs Predicted Close Prices')

# --- SHAP Interpretation for XGBoost ---
print('\nGenerating SHAP summary plot for XGBoost...')
# Use the training data for the explainer for reference
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train_diff)
# Generate and show plot
plt.title('SHAP Feature Importance for XGBoost')
shap.summary_plot(shap_values, X_train_diff, plot_type="bar", show=False)
plt.show()

# LSTM (use scaled data)
time_steps = 10
X_train_lstm = X_train.reset_index(drop=True)
y_train_lstm = y_train.reset_index(drop=True)
X_test_lstm = X_test.reset_index(drop=True)
y_test_lstm = y_test.reset_index(drop=True)

# Scale features and target for LSTM
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_lstm)
X_test_scaled = scaler_X.transform(X_test_lstm)
y_train_scaled = scaler_y.fit_transform(y_train_lstm.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test_lstm.values.reshape(-1, 1)).flatten()

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
y_train_scaled_series = pd.Series(y_train_scaled)
y_test_scaled_series = pd.Series(y_test_scaled)

y_pred_lstm_scaled, lstm_model = train_predict_lstm(X_train_scaled_df, y_train_scaled_series, X_test_scaled_df, time_steps=time_steps, epochs=50)
# LSTM predictions are shorter due to time_steps offset
dates_test_lstm = dates_test.iloc[time_steps:].reset_index(drop=True)
y_test_lstm_eval = y_test_scaled_series.iloc[time_steps:].reset_index(drop=True)
# Inverse transform predictions and true values
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_test_lstm_eval_inv = scaler_y.inverse_transform(y_test_lstm_eval.values.reshape(-1, 1)).flatten()
metrics_lstm = regression_metrics(y_test_lstm_eval_inv, y_pred_lstm)
print('\nLSTM Performance (with scaling):')
for k, v in metrics_lstm.items():
    print(f'{k}: {v:.4f}')
plot_actual_vs_predicted(dates_test_lstm, y_test_lstm_eval_inv, y_pred_lstm, title='LSTM: Actual vs Predicted Close Prices (Scaled)')

# Debug: Print min/max of y_train, y_test, and their scaled versions
print('\n[DEBUG] y_train min/max:', y_train.min(), y_train.max())
print('[DEBUG] y_test min/max:', y_test.min(), y_test.max())
print('[DEBUG] y_train_scaled min/max:', y_train_scaled.min(), y_train_scaled.max())
print('[DEBUG] y_test_scaled min/max:', y_test_scaled.min(), y_test_scaled.max())

# Print last 10 actual Close values from test set
print('\n[DEBUG] Last 10 actual Close values:', list(y_test.tail(10)))
# Print last 10 LSTM single-step predictions (not multi-step)
print('[DEBUG] Last 10 LSTM single-step predictions:', list(y_pred_lstm[-10:]))

# --- Multi-step Forecasting (Next 7 Days) ---
forecast_horizon = 7
print(f'\nMulti-step Forecasting (Next {forecast_horizon} Days):')

# Get the last row of features from the test set
last_features = X_test.iloc[[-1]].copy()

# --- Linear Regression (absolute) ---
preds_lr = []
current_features_lr = last_features.copy()
# The 'Close' price isn't in X_test, so we can't update it. 
# We will assume a simple feature update for the sake of forecasting.
for _ in range(forecast_horizon):
    pred = model.predict(current_features_lr)[0]
    preds_lr.append(pred)
    # Note: For a pure autoregressive model, we would update features based on the new 'pred'.
    # Since we can't do that perfectly here, this forecast might be simplistic.
print('Linear Regression 7-day forecast:', preds_lr)

# --- Random Forest (differenced) ---
preds_rf = []
last_close = y_test.iloc[-1]
current_features_rf = last_features.copy()
for _ in range(forecast_horizon):
    pred_diff = rf_model.predict(current_features_rf)[0]
    new_close = last_close + pred_diff
    preds_rf.append(new_close)
    last_close = new_close
print('Random Forest 7-day forecast:', preds_rf)

# --- XGBoost (differenced) ---
preds_xgb = []
last_close = y_test.iloc[-1]
current_features_xgb = last_features.copy()
for _ in range(forecast_horizon):
    pred_diff = xgb_model.predict(current_features_xgb)[0]
    new_close = last_close + pred_diff
    preds_xgb.append(new_close)
    last_close = new_close
print('XGBoost 7-day forecast:', preds_xgb)

# LSTM (recursive, absolute)
last_known_X_lstm = X_train_scaled_df.iloc[-time_steps:]
lstm_preds = recursive_multi_step_lstm(lstm_model, last_known_X_lstm, forecast_horizon, time_steps, scaler_X, scaler_y, X_test.columns.tolist())
print('LSTM 7-day forecast:', lstm_preds)

# Convert forecasts to Python floats for plotting
preds_lr = [float(x) for x in preds_lr]
preds_rf = [float(x) for x in preds_rf]
preds_xgb = [float(x) for x in preds_xgb]
lstm_preds = [float(x) for x in lstm_preds]

# Print forecast values and dates for verification
print('Forecast Dates:', pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:])
print('Linear Regression:', preds_lr)
print('Random Forest:', preds_rf)
print('XGBoost:', preds_xgb)
print('LSTM:', lstm_preds)

# Debug: Print shapes and types before plotting
print('\n[DEBUG] dates_future:', pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:])
print('[DEBUG] preds_lr:', np.array(preds_lr).shape, type(preds_lr[0]) if len(preds_lr) > 0 else None)
print('[DEBUG] preds_rf:', np.array(preds_rf).shape, type(preds_rf[0]) if len(preds_rf) > 0 else None)
print('[DEBUG] preds_xgb:', np.array(preds_xgb).shape, type(preds_xgb[0]) if len(preds_xgb) > 0 else None)
print('[DEBUG] lstm_preds:', np.array(lstm_preds).shape, type(lstm_preds[0]) if len(lstm_preds) > 0 else None)

# Convert all forecast arrays to numpy float arrays
preds_lr = np.array(preds_lr, dtype=float)
preds_rf = np.array(preds_rf, dtype=float)
preds_xgb = np.array(preds_xgb, dtype=float)
lstm_preds = np.array(lstm_preds, dtype=float)

# Align Prophet forecast to dates_future for fair comparison
prophet_forecast, prophet_model = train_predict_prophet(df, forecast_horizon=7)
prophet_plot = prophet_forecast[prophet_forecast['ds'].isin(pd.to_datetime(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:]))]
if len(prophet_plot) < len(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:]):
    print('[WARNING] Not all Prophet forecast dates match dates_future. Interpolating...')
    # Interpolate Prophet's yhat to match dates_future
    prophet_plot = prophet_forecast.set_index('ds').reindex(pd.to_datetime(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:])).interpolate(method='time').reset_index()
    prophet_plot.rename(columns={'index': 'ds'}, inplace=True)

# Plot all forecasts with aligned Prophet
plt.figure(figsize=(10,6))
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], preds_lr, label='Linear Regression', marker='o')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], preds_rf, label='Random Forest', marker='o')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], preds_xgb, label='XGBoost', marker='o')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], lstm_preds, label='LSTM', marker='o')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], prophet_plot['yhat'], label='Prophet', marker='o')
plt.title('7-Day Multi-step Forecasts (All Models)')
plt.xlabel('Date')
plt.ylabel('Predicted Close Price')
plt.legend()
plt.grid(True)
plt.show()

# --- Direct Multi-step (Seq2Seq) LSTM/GRU Forecasting ---
time_steps_seq2seq = 20
output_steps = 7
print(f'\nDirect Seq2Seq LSTM/GRU Forecasting (Next {output_steps} Days):')
# Prepare scaled data for seq2seq
X_train_seq2seq = X_train_scaled_df
X_test_seq2seq = X_test_scaled_df
y_train_seq2seq = y_train_scaled_series
y_test_seq2seq = y_test_scaled_series
# Train seq2seq LSTM (bidirectional, can switch to GRU)
y_pred_seq2seq_scaled, seq2seq_model = train_predict_seq2seq(
    X_train_seq2seq, y_train_seq2seq, X_test_seq2seq,
    time_steps=time_steps_seq2seq, output_steps=output_steps, epochs=50, use_gru=False, use_bidir=True)
# Take the last prediction (most recent window)
y_pred_seq2seq_scaled_last = y_pred_seq2seq_scaled[-1]
y_pred_seq2seq = scaler_y.inverse_transform(y_pred_seq2seq_scaled_last.reshape(-1, 1)).flatten()
print('Seq2Seq LSTM 7-day forecast:', y_pred_seq2seq)

# --- Prophet Forecasting ---
print('\nProphet Forecasting (Next 7 Days):')
prophet_forecast, prophet_model = train_predict_prophet(df, forecast_horizon=7)
print('Prophet 7-day forecast:', list(prophet_forecast['yhat']))
# Plot Prophet forecast with other models
plt.figure(figsize=(12, 7))
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], preds_lr, label='Linear Regression', marker='o', linestyle='--')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], preds_rf, label='Random Forest', marker='o', linestyle='--')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], preds_xgb, label='XGBoost', marker='o', linestyle='--')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], lstm_preds, label='LSTM (Recursive)', marker='o', linestyle='--')
plt.plot(pd.date_range(start=dates_test.iloc[-1], periods=forecast_horizon+1)[1:], y_pred_seq2seq, label='Seq2Seq LSTM (Direct)', marker='o', linestyle='-')
plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet', marker='o', linestyle='-')
plt.title('7-Day Multi-step Forecasts (All Models)')
plt.xlabel('Date')
plt.ylabel('Predicted Close Price')
plt.legend()
plt.grid(True)
plt.show() 