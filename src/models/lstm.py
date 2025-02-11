import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_residual_lstm_model(input_shape):
    """
    创建带有残差连接的LSTM模型，增强正则化以防止过拟合
    Args:
        input_shape: 输入形状 (sequence_length, features)
    """
    inputs = tf.keras.Input(shape=(input_shape[0], 2))
    
    # 添加BatchNormalization来标准化输入
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # 第一个LSTM块 - 减少单元数并添加正则化
    x = tf.keras.layers.LSTM(
        32,  # 减少单元数
        return_sequences=True,
        dropout=0.3,  # 增加dropout
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),  # 添加L1L2正则化
    )(x)
    
    # 残差连接1
    residual = tf.keras.layers.Dense(
        32,
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(inputs)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # 额外的dropout
    
    # 第二个LSTM块 - 同样添加正则化
    lstm_out = tf.keras.layers.LSTM(
        16,  # 进一步减少单元数
        return_sequences=True,
        dropout=0.3,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
    )(x)
    
    # 残差连接2
    residual = tf.keras.layers.Dense(
        16,
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)
    x = tf.keras.layers.Add()([lstm_out, residual])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # 减少全连接层的复杂度
    x = tf.keras.layers.Dense(
        8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # 输出层
    outputs = tf.keras.layers.Dense(
        1,
        kernel_regularizer=tf.keras.regularizers.L2(1e-4)
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_model_with_config(config):
    """创建并编译模型"""
    # 使用新的残差LSTM模型
    model = create_residual_lstm_model((config["input_length"], 1))
    
    # 使用带有学习率调度的优化器
    initial_learning_rate = config.get("learning_rate", 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )
    return model

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