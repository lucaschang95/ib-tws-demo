import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# https://www.linkedin.com/pulse/understanding-lstm-python-examples-tensorflow-keras-rany-7gckc
print("Starting LSTM demo...")

# Generate sample data with linear pattern plus noise
# 增加基础数据长度到100，以便获得99个输入点和1个输出点
base_data = np.array([range(i, i + 10) for i in range(100)])  # 修改为100个连续数

# noise = np.random.normal(0, 0.1, base_data.shape)
data = base_data
# data = base_data + noise

# 修改切片以使用99个输入点
X, y = data[:, :-1], data[:, -1]

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 定义简单模型
simple_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(9, 1)),
    tf.keras.layers.Dense(1)
])

simple_model.compile(optimizer='adam', loss='mse')

# 定义复杂模型
complex_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', input_shape=(9, 1), 
                        return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 修改编译参数
complex_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 训练简单模型
print("\nTraining simple model...")
simple_history = simple_model.fit(X, y, epochs=200, verbose=0)

# 训练复杂模型
print("\nTraining complex model...")
# 添加早停和学习率调整
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)
complex_history = complex_model.fit(
    X, y,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler],
    verbose=0
)

# 测试两个模型
test_input = np.array(range(50, 149))  # 生成99个连续数字
# test_input = np.array(range(120, 219))  # 生成99个连续数字
test_input = test_input.reshape((1, 99, 1))

simple_prediction = simple_model.predict(test_input, verbose=0)
complex_prediction = complex_model.predict(test_input, verbose=0)

print("\nModel Comparison:")
print(f'Simple model predicted value: {simple_prediction[0][0]}')
print(f'Complex model predicted value: {complex_prediction[0][0]}')

# 计算并比较两个模型的MSE
simple_mse = simple_model.evaluate(X, y, verbose=0)
complex_mse = complex_model.evaluate(X, y, verbose=0)[0]  # 第一个值是MSE，第二个是MAE

print(f'\nSimple model MSE: {simple_mse}')
print(f'Complex model MSE: {complex_mse}')

# 可视化训练过程（如果您想看训练过程的图表，取消下面的注释）
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(simple_history.history['loss'], label='Simple Model Loss')
plt.title('Simple Model Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(complex_history.history['loss'], label='Training Loss')
plt.plot(complex_history.history['val_loss'], label='Validation Loss')
plt.title('Complex Model Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
"""  