import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from keras.callbacks import EarlyStopping
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

def create_lstm_dataset(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_predict_lstm(X_train, y_train, X_test, time_steps=10, epochs=50, batch_size=32, validation_split=0.1):
    X_train_lstm, y_train_lstm = create_lstm_dataset(X_train, y_train, time_steps)
    X_test_lstm, _ = create_lstm_dataset(X_test, y_train, time_steps) 
    n_features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(time_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=validation_split, callbacks=[es])
    y_pred = model.predict(X_test_lstm).flatten()
    return y_pred, model

def recursive_multi_step_lstm(model, last_known_X, n_steps, time_steps, scaler_X, scaler_y, feature_cols):
    preds = []
    current_X = last_known_X.copy()
    for _ in range(n_steps):
        X_input = scaler_X.transform(current_X[feature_cols].values)
        X_input = X_input.reshape(1, time_steps, len(feature_cols))
        y_pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        preds.append(y_pred)
        new_row_df = pd.DataFrame([current_X.iloc[-1]], columns=feature_cols)
        new_row_df['Close'] = y_pred 
        temp_df = pd.concat([current_X, new_row_df], ignore_index=True)
        temp_df = add_technical_indicators(temp_df)
        new_row = temp_df.iloc[-1].ffill().bfill()
        current_X = pd.concat([current_X.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
    return preds

def create_seq2seq_dataset(X, y, time_steps=20, output_steps=7):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - output_steps + 1):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[(i + time_steps):(i + time_steps + output_steps)].values)
    return np.array(Xs), np.array(ys)

def train_predict_seq2seq(X_train, y_train, X_test, time_steps=20, output_steps=7, epochs=50, batch_size=32, validation_split=0.1, use_gru=False, use_bidir=True):
    X_train_seq, y_train_seq = create_seq2seq_dataset(X_train, y_train, time_steps, output_steps)
    X_test_seq, _ = create_seq2seq_dataset(X_test, y_train, time_steps, output_steps)
    n_features = X_train.shape[1]
    model = Sequential()
    RNNLayer = GRU if use_gru else LSTM
    if use_bidir:
        model.add(Bidirectional(RNNLayer(100, activation='relu', return_sequences=True), input_shape=(time_steps, n_features)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(RNNLayer(100, activation='relu')))
        model.add(Dropout(0.2))
    else:
        model.add(RNNLayer(100, activation='relu', return_sequences=True, input_shape=(time_steps, n_features)))
        model.add(Dropout(0.2))
        model.add(RNNLayer(100, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=validation_split, callbacks=[es])
    y_pred = model.predict(X_test_seq)
    return y_pred, model

def train_predict_prophet(df, forecast_horizon=7):
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_horizon)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_horizon), model 