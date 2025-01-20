import numpy as np

def generate_sequence_data(sequence_length, input_length, max_start_point):
    """Generate sample data with linear pattern"""
    # Generate sequences starting from different points to cover a wider range
    start_points = np.random.randint(1, max_start_point, sequence_length)  # Generate random starting points from 1 to max_start_point
    base_data = np.array([range(int(i), int(i) + input_length + 1) for i in start_points])
    noise = np.random.normal(0, 0.1, base_data.shape)
    return base_data + noise

def prepare_data(data):
    """Prepare data for LSTM input"""
    X, y = data[:, :-1], data[:, -1]
    # Reshape input to be [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y
