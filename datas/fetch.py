import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
import pandas as pd

from utils.args import load_args
from utils.logger import setup_logger
from utils.config import load_config
from ib.client import IBClient
from utils.preprocess import preprocess
from models.lstm import create_model_with_config
from ib_async import *
from models.trainer import LSTMTrainer
from process import check_missing_data, merge_csv_files

def convert_bardata_to_df(bar_data_list):
    """Convert list of BarData objects to pandas DataFrame"""
    data = []
    for bar in bar_data_list:
        data.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'average': bar.average,
            'barCount': bar.barCount
        })
    return pd.DataFrame(data)

def main():
    setup_logger(clear_log=True)
    args = load_args()

    config = load_config(args.config)

    client = IBClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id
    )
    client.connect()
    contract = Stock('AAPL', 'SMART', 'USD')

    # Read the existing data file
    df = pd.read_csv('test-data/aapl_bars.csv')
    missing_dates = check_missing_data(df)
    
    fetched_bars = client.get_real_historical_data(contract, '2025-01-09', '2025-02-10')

    # Convert BarData objects to DataFrame
    fetched_df = convert_bardata_to_df(fetched_bars)

    # Now merge with existing CSV data if the file exists
    csv_files = ['test-data/aapl_bars.csv']
    
    existing_files = []
    for f in csv_files:
        if os.path.exists(f):
            existing_files.append(f)
        else:
            print(f"Warning: File {f} does not exist")
    
    if existing_files:
        merged_data = merge_csv_files(fetched_df, existing_files)
    else:
        merged_data = fetched_df

    # Save to CSV
    merged_data.to_csv('test-data/aapl_bars14.csv', index=True)
    
    # Disconnect
    client.disconnect()

if __name__ == "__main__":
    main()
