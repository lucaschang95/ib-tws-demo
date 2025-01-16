import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# https://www.linkedin.com/pulse/understanding-lstm-python-examples-tensorflow-keras-rany-7gckc
print("Starting LSTM demo...")

# Generate sample data
data = np.array([range(i, i + 10) for i in range(100)])
X, y = data[:, :-1], data[:, -1]
print(X)


# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(9, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X, y, epochs=200, verbose=1)

# Make a prediction
test_input = np.array([120, 121, 122, 123, 124, 125, 126, 127, 128])
test_input = test_input.reshape((1, 9, 1))
predicted_value = model.predict(test_input, verbose=0)
print(f'Predicted value: {predicted_value[0][0]}')  