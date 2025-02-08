import pandas as pd
from typing import List, Optional
import numpy as np

def clean_missing_values(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    method: str = 'ffill'
) -> pd.DataFrame:
    if columns is None:
        return df.fillna(method=method)
    else:
        df_clean = df.copy()
        df_clean[columns] = df_clean[columns].fillna(method=method)
        return df_clean

def remove_outliers(df, column, n_sigmas=3):
    """移除异常值"""
    mean = df[column].mean()
    std = df[column].std()
    return df[abs(df[column] - mean) <= n_sigmas * std]

def normalize_data(df, config):
    """
    标准化数据并保存缩放参数
    """
    columns = ['close', 'volume']
    df_norm = df.copy()
    
    # 计算并保存缩放参数
    close_mean = df['close'].mean()
    close_std = df['close'].std()
    config['close_mean'] = close_mean
    config['close_std'] = close_std
    
    # 标准化收盘价
    df_norm['close'] = (df['close'] - close_mean) / close_std
    
    # 对成交量进行对数标准化
    df_norm['volume'] = np.log1p(df['volume'])
    volume_mean = df_norm['volume'].mean()
    volume_std = df_norm['volume'].std()
    config['volume_mean'] = volume_mean
    config['volume_std'] = volume_std
    df_norm['volume'] = (df_norm['volume'] - volume_mean) / volume_std
    
    return df_norm

def split(df, train_ratio=0.7, val_ratio=0.2):
    """
    按时间顺序划分数据集为训练集、验证集和测试集
    
    Args:
        df: pandas DataFrame，原始数据
        train_ratio: float，训练集比例
        val_ratio: float，验证集比例
        
    Returns:
        train_df: pandas DataFrame，训练集
        val_df: pandas DataFrame，验证集 
        test_df: pandas DataFrame，测试集
    """
    n_points = len(df)
    train_size = int(n_points * train_ratio)
    val_size = int(n_points * val_ratio)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:(train_size + val_size)].copy()
    test_df = df.iloc[(train_size + val_size):].copy()

    return train_df, val_df, test_df

def create_sequences(data, sequence_length):
    """
    创建用于时间序列预测的输入序列和目标值
    
    Args:
        data: pandas DataFrame，包含多个特征的原始数据
        sequence_length: int，输入序列长度
        
    Returns:
        X: 输入序列数组
        y: 目标值数组
    """
    features = ['close', 'volume']
    X = []
    y = []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[features].iloc[i:(i + sequence_length)].values)
        y.append(data['close'].iloc[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_data(X):
    """
    重塑数据为LSTM模型所需的3D格式
    
    Args:
        X: 输入数据数组
        
    Returns:
        重塑后的数组，形状为(样本数, 时间步长, 特征数)
    """
    # X已经是3D格式 (samples, sequence_length, features)，无需reshape
    return X

def preprocess(df, config):
    df = normalize_data(df, config)
    # 划分数据集
    train_data, val_data, test_data = split(df)

    # print('train_data:')
    # print(train_data.head())
    # print('val_data:')
    # print(val_data.head())
    # print('test_data:')
    # print(test_data.head())

    # 训练集
    X_train, y_train = create_sequences(
        train_data, sequence_length=config["input_length"]
    )

    # 验证集
    X_val, y_val = create_sequences(val_data, sequence_length=config["input_length"])

    # 测试集
    X_test, y_test = create_sequences(test_data, sequence_length=config["input_length"])

    return X_train, y_train, X_val, y_val, X_test, y_test