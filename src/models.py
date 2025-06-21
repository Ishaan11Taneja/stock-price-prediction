import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from src.feature_engineering import add_technical_indicators

def train_predict_rf(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

def train_predict_xgb(X_train, y_train, X_test):
    model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model

def train_predict_prophet(df, forecast_horizon=7):
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_horizon)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_horizon), model 