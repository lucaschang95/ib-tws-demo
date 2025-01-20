import tensorflow as tf

def create_simple_model(input_shape):
    """Create a simple LSTM model
    
    Args:
        input_shape: Tuple specifying the input shape (sequence_length, features)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
