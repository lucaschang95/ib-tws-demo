import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PrecisionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        metrics_str = []
        for metric, value in logs.items():
            metrics_str.append(f"{metric}: {value:.6f}")
        print(f"\nEpoch {epoch + 1}: {' - '.join(metrics_str)}")

def get_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        min_delta=1e-4,
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-6
    )

    return [early_stopping, lr_scheduler]

def train_models(simple_model, X_train, y_train, X_val, y_val, epochs=200):
    callbacks = get_callbacks()
    
    simple_history = simple_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    return simple_history

def create_simple_model(input_shape):
    inputs = tf.keras.Input(shape=(input_shape[0], 2))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_model_with_config(config):
    """创建并编译模型"""
    simple_model = create_simple_model((config["input_length"], 1))
    simple_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )
    return simple_model

def train_and_save_model(model, X_train, y_train, X_val, y_val, config):
    """训练模型并保存"""
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'lstm_model.keras')
    
    simple_history = train_models(
        model, X_train, y_train, X_val, y_val, epochs=config["epochs"]
    )
    
    model.save(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model, simple_history

# 添加以下代码（放在import之后，函数定义之前）
plt.rcParams['font.sans-serif'] = ['PingFang']  # Mac系统
# 如果是Windows系统，使用：
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_mse(model, X, y):
    """计算模型在给定数据集上的MSE"""
    return model.evaluate(X, y, verbose=0)

def evaluate_models(simple_model, simple_history, train_gen=None, val_gen=None, test_gen=None):
    # 如果有训练历史，直接从历史记录中获取最后的损失值
    if simple_history is not None:
        if 'loss' in simple_history.history:
            logging.info(f'Training MSE: {simple_history.history["loss"][-1]}')
        if 'val_loss' in simple_history.history:
            logging.info(f'Validation MSE: {simple_history.history["val_loss"][-1]}')
    else:
        # 如果没有历史记录，使用生成器评估
        if train_gen is not None:
            train_loss = simple_model.evaluate(train_gen, verbose=0)
            logging.info(f'Training MSE: {train_loss}')
        
        if val_gen is not None:
            val_loss = simple_model.evaluate(val_gen, verbose=0)
            logging.info(f'Validation MSE: {val_loss}')
    
    # 测试集的MSE
    if test_gen is not None:
        test_loss = simple_model.evaluate(test_gen, verbose=0)
        logging.info(f'Test MSE: {test_loss}')