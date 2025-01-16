import numpy as np
import pandas as pd
import tensorflow as tf

def load_and_prepare_data(file_path, sequence_length=10):
    print(f"Loading data from {file_path}")
    print(f"Using sequence length: {sequence_length}")
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 使用收盘价作为特征
    data = df['close'].values
    
    # 数据标准化
    mean = np.mean(data)
    std = np.std(data)
    data_normalized = (data - mean) / std
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i:(i + sequence_length)])
        y.append(data_normalized[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # 重塑数据以适应 LSTM 输入格式 [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 分割训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, mean, std

def create_model(sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def predict_next_day(model, last_sequence, mean, std):
    """
    使用训练好的模型预测下一天的股票价格
    
    Args:
        model: 训练好的LSTM模型
        last_sequence: 最后N天的收盘价序列（N等于训练时使用的sequence_length）
        mean: 用于标准化的均值
        std: 用于标准化的标准差
    
    Returns:
        float: 预测的下一天收盘价
    """
    # 标准化输入数据
    last_sequence_normalized = (last_sequence - mean) / std
    
    # 重塑数据以适应LSTM输入格式 [samples, time steps, features]
    X_pred = np.reshape(last_sequence_normalized, (1, len(last_sequence), 1))
    
    # 预测
    pred_normalized = model.predict(X_pred, verbose=0)
    
    # 反标准化得到实际价格
    predicted_price = (pred_normalized[0][0] * std) + mean
    
    return predicted_price

def train_and_evaluate():
    # 设置参数
    sequence_length = 10
    epochs = 50
    batch_size = 32
    
    # 准备数据
    X_train, X_test, y_train, y_test, mean, std = load_and_prepare_data(
        'test-data/aapl_bars4.csv', 
        sequence_length
    )
    
    # 创建和训练模型
    model = create_model(sequence_length)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    # 评估模型
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 反标准化预测结果
    y_pred = (y_pred * std) + mean
    y_test_actual = (y_test * std) + mean
    
    # 计算预测误差
    mape = np.mean(np.abs((y_test_actual - y_pred.flatten()) / y_test_actual)) * 100
    print(f'Mean Absolute Percentage Error: {mape:.2f}%')
    
    # 获取最后一个序列用于预测
    last_sequence = X_test[-1].flatten()  # 获取最后一个测试序列
    next_day_price = predict_next_day(model, last_sequence, mean, std)
    print(f'\n预测下一天的收盘价: ${next_day_price:.2f}')
    
    return model, history, y_pred, y_test_actual, next_day_price

if __name__ == '__main__':
    model, history, predictions, actual, tomorrow_price = train_and_evaluate()
    print(f'\n最终预测结果：明天的预期收盘价是 ${tomorrow_price:.2f}') 