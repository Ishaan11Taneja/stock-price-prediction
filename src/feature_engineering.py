import pandas as pd
import numpy as np

def add_technical_indicators(df):
    df = df.copy()
    # Simple Moving Average
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    # Exponential Moving Average
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df 