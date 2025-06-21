import os
import pandas as pd

def load_tata_global_data(csv_path=None):
    """
    Loads NSE TATA GLOBAL stock data from a CSV file. If csv_path is None, looks for data/raw/tata_global.csv.
    Returns a pandas DataFrame.
    """
    if csv_path is None:
        csv_path = os.path.join('data', 'raw', 'tata_global.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}. Please download the NSE TATA GLOBAL dataset and place it there.")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)  # Sort by date
    return df 