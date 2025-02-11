import pandas as pd
from typing import List, Optional
import numpy as np
import tensorflow as tf

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

def split(df, train_ratio=0.8, val_ratio=0.1):
    """
    按时间顺序划分数据集为训练集、验证集和测试集
    
    Args:
        df: pandas DataFrame，原始数据
        train_ratio: float，训练集比例，默认0.8
        val_ratio: float，验证集比例，默认0.1
        
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
    创建用于时间序列预测的输入序列和目标值，使用生成器来优化内存使用
    
    Args:
        data: pandas DataFrame，包含多个特征的原始数据
        sequence_length: int，输入序列长度
        
    Returns:
        X: 输入序列数组
        y: 目标值数组
    """
    features = ['close', 'volume']
    total_sequences = len(data) - sequence_length - 1
    
    # 预分配数组以避免动态增长
    X = np.zeros((total_sequences, sequence_length, len(features)))
    y = np.zeros(total_sequences)
    
    # 使用更高效的数组操作
    for i in range(total_sequences):
        X[i] = data[features].iloc[i:(i + sequence_length)].values
        y[i] = data['close'].iloc[i + sequence_length]
    
    return X, y

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

class TimeSeriesGenerator(tf.keras.utils.PyDataset):
    """
    时间序列数据生成器，用于批量生成训练数据
    继承自tf.keras.utils.PyDataset以确保与Keras训练接口兼容
    """
    def __init__(self, data, sequence_length, batch_size=32, features=['close', 'volume']):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.features = features
        self.n_features = len(features)
        self.total_sequences = len(data) - sequence_length - 1
        self.indexes = np.arange(self.total_sequences)

    def __len__(self):
        """返回每个epoch的批次数"""
        return int(np.ceil(self.total_sequences / self.batch_size))

    def __getitem__(self, idx):
        """获取一个批次的数据"""
        if isinstance(idx, slice):
            # 处理切片操作
            indices = range(*idx.indices(len(self)))
            return [self._get_single_item(i) for i in indices]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        """获取单个批次的数据"""
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.total_sequences)
        batch_indexes = self.indexes[start_idx:end_idx]

        X = np.zeros((len(batch_indexes), self.sequence_length, self.n_features))
        y = np.zeros(len(batch_indexes))

        for i, seq_idx in enumerate(batch_indexes):
            X[i] = self.data[self.features].iloc[seq_idx:(seq_idx + self.sequence_length)].values
            y[i] = self.data['close'].iloc[seq_idx + self.sequence_length]

        return X, y

    # def on_epoch_end(self):
    #     """每个epoch结束时重新打乱数据"""
    #     np.random.shuffle(self.indexes)

def preprocess(df, config):
    df = normalize_data(df, config)
    # 划分数据集
    train_data, val_data, test_data = split(df)

    # 创建数据生成器
    train_gen = TimeSeriesGenerator(
        train_data, 
        sequence_length=config["input_length"],
        batch_size=config.get("batch_size", 32)
    )
    val_gen = TimeSeriesGenerator(
        val_data, 
        sequence_length=config["input_length"],
        batch_size=config.get("batch_size", 32)
    )
    test_gen = TimeSeriesGenerator(
        test_data, 
        sequence_length=config["input_length"],
        batch_size=config.get("batch_size", 32)
    )

    return train_gen, val_gen, test_gen