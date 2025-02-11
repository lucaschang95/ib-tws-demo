import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_simple_model(input_shape):
    inputs = tf.keras.Input(shape=(input_shape[0], 2))
    x = tf.keras.layers.LSTM(
        64,
        return_sequences=True, 
        dropout=0.2
    )(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)  # Dense层之前的dropout
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # 输出层之前的dropout
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