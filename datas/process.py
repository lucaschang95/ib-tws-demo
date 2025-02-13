import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import pandas as pd
from typing import List
from utils.holidays import get_market_holidays

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from utils.args import load_args
from utils.logger import setup_logger
from utils.config import load_config
from ib.client import IBClient
from utils.preprocess import preprocess
from models.lstm import create_model_with_config
from ib_async import *
from models.trainer import LSTMTrainer


def merge_csv_files(existing_data: pd.DataFrame, file_paths: List[str], output_path: str = 'merged_data.csv') -> pd.DataFrame:
    # 存储所有数据框的列表，首先添加现有数据
    dfs = []
    if isinstance(existing_data, pd.DataFrame) and not existing_data.empty:
        dfs.append(existing_data)
    elif isinstance(existing_data, list) and existing_data:
        dfs.extend(existing_data)
    
    # 读取所有 CSV 文件
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # 删除第一列（如果是索引列）
            if df.columns[0] == 'Unnamed: 0':
                df = df.drop(df.columns[0], axis=1)
            dfs.append(df)
        else:
            print(f"Warning: File {file_path} does not exist")
    
    if not dfs:
        raise ValueError("No valid data to merge")
    
    # 合并所有数据框
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 去除重复行
    merged_df = merged_df.drop_duplicates()
    
    # 确保日期列是日期时间格式
    date_column = 'date' if 'date' in merged_df.columns else merged_df.columns[0]
    merged_df[date_column] = pd.to_datetime(merged_df[date_column])
    
    # 按日期排序
    merged_df = merged_df.sort_values(by=date_column)
    
    # 重置索引并保存到文件（保存索引列）
    merged_df = merged_df.reset_index(drop=True)  # 确保索引从0开始
    
    return merged_df

def check_missing_data(data):
    """
    Check for missing trading days in stock market data
    Args:
        data: DataFrame containing stock data with 'date' column
    Returns:
        List of missing trading days
    """
    # Get unique dates as strings (date portion only)
    unique_dates = data['date'].str.split(' ').str[0].unique()
    # Convert to datetime
    unique_dates = pd.to_datetime(unique_dates)
    
    # Create a complete range of business days
    date_range = pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq='B')
    
    # Get all holidays for the date range
    all_holidays = []
    years = range(unique_dates.min().year, unique_dates.max().year + 1)
    for year in years:
        all_holidays.extend(get_market_holidays(year))
    print('all_holidays', all_holidays)
    
    # Convert unique_dates to set for faster lookup
    unique_dates_set = set(unique_dates.date)
    holidays_set = set(holiday.date() for holiday in all_holidays)
    
    # Find missing business days (excluding holidays)
    missing_dates = []
    for date in date_range:
        if date.date() not in unique_dates_set and date.date() not in holidays_set:
            missing_dates.append(date)
    
    if missing_dates:
        print("\nMissing data found for the following trading days (excluding holidays):")
        for date in sorted(missing_dates):
            print(f"Missing data on: {date.date()}")
        print(f"\nTotal missing trading days: {len(missing_dates)}")
    else:
        print("\nNo missing trading days found in the date range")
    
    return missing_dates