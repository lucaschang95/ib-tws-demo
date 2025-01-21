import tensorflow as tf

def create_simple_model(input_shape):
    """Create a simple LSTM model
    
    Args:
        input_shape: Tuple specifying the input shape (sequence_length, features)
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
