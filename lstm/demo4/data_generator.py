import numpy as np

def generate_sine_wave(n_points, amplitude, noise_scale=0.25):
    frequency = 0.02
    time = np.arange(n_points)
    print("\nTime string generated, first 5 time points:", time[:5])
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    noise = np.random.normal(0, amplitude * noise_scale, len(time))
    return time, sine_wave + noise

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_data(X):
    return X.reshape((X.shape[0], X.shape[1], 1))

def normalize_data(wave_data):
    min_val = np.min(wave_data)
    max_val = np.max(wave_data)
    normalized_data = (wave_data - min_val) / (max_val - min_val)
    return normalized_data