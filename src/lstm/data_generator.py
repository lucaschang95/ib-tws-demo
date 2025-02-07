import logging
import numpy as np

def generate_time_series(n_points):
    """生成时间序列。"""
    return np.arange(n_points)

def generate_noise(time, amplitude, noise_scale=0.25):
    """生成噪声数据。"""
    return np.random.normal(0, amplitude * noise_scale, len(time))

def generate_sine_wave(n_points, amplitude, noise_scale=0.25):
    """生成带噪声的正弦波数据。"""
    time = generate_time_series(n_points)
    logging.info(f"Time string generated, first 5 time points: {time[:5]}")
    frequency = 0.02
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    noise = generate_noise(time, amplitude, noise_scale)
    return time, sine_wave + noise

def prepare_data(X):
    return X.reshape((X.shape[0], X.shape[1], 1))

def split_data(data, train_ratio=0.7, val_ratio=0.2):
    """按时间顺序划分数据集为训练集、验证集和测试集。"""
    n_points = len(data)
    train_size = int(n_points * train_ratio)
    val_size = int(n_points * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:(train_size + val_size)]
    test_data = data[(train_size + val_size):]

    logging.info(f"Train set size: {len(train_data)}")
    logging.info(f"Validation set size: {len(val_data)}")
    logging.info(f"Test set size: {len(test_data)}")

    return train_data, val_data, test_data

def normalize_data(wave_data):
    min_val = np.min(wave_data)
    max_val = np.max(wave_data)
    normalized_data = (wave_data - min_val) / (max_val - min_val)
    return normalized_data

def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

