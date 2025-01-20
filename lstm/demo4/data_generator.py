import numpy as np

def generate_sine_wave(n_points, amplitude=10):
    frequency = 0.1
    time = np.arange(0, n_points, 0.1)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    noise = np.random.normal(0, amplitude/4, len(time))
    print(noise)
    np.savetxt('noise.csv', noise, delimiter=',')
    res = sine_wave + noise
    np.savetxt('res.csv', res, delimiter=',')
    return time, res

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_data(X):
    """Reshape input for LSTM [samples, time steps, features]"""
    return X.reshape((X.shape[0], X.shape[1], 1))
