import logging
import numpy as np
import matplotlib.pyplot as plt

# 添加以下代码（放在import之后，函数定义之前）
plt.rcParams['font.sans-serif'] = ['PingFang']  # Mac系统
# 如果是Windows系统，使用：
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_mse(model, X, y):
    """计算模型在给定数据集上的MSE"""
    return model.evaluate(X, y, verbose=0)

def evaluate_models(simple_model, simple_history, X_train=None, y_train=None, X_val=None, y_val=None, X_test = None, y_test = None):
    # 如果有训练历史，直接从历史记录中获取最后的损失值
    if simple_history is not None:
        if 'loss' in simple_history.history:
            logging.info(f'Training MSE: {simple_history.history["loss"][-1]}')
        if 'val_loss' in simple_history.history:
            logging.info(f'Validation MSE: {simple_history.history["val_loss"][-1]}')
    else:
        # 如果没有历史记录，才需要重新计算
        if X_train is not None and y_train is not None:
            train_mse = calculate_mse(simple_model, X_train, y_train)
            logging.info(f'Training MSE: {train_mse}')
        
        if X_val is not None and y_val is not None:
            val_mse = calculate_mse(simple_model, X_val, y_val)
            logging.info(f'Validation MSE: {val_mse}')
    
    # 测试集的MSE还是需要计算，因为测试集在训练时没有使用过
    test_mse = calculate_mse(simple_model, X_test, y_test)
    logging.info(f'Test MSE: {test_mse}')