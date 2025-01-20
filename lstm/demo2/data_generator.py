import numpy as np

def generate_sequence_data(sequence_length=100, input_length=9):
    """Generate sample data with linear pattern"""
    base_data = np.array([range(i, i + input_length + 1) for i in range(sequence_length)])
    return base_data

def prepare_data(data):
    """Prepare data for LSTM input"""
    X, y = data[:, :-1], data[:, -1]
    # Reshape input to be [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y
