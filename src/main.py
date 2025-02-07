from utils.args import load_args
from utils.logger import setup_logger
from utils.config import load_config
from ib.client import IBClient
from lstm.models import create_simple_model
from lstm.train import train_models
from lstm.evaluate import evaluate_models
import asyncio
import os
import tensorflow as tf

from utils.preprocess import preprocess

def create_and_train_model(X_train, y_train, X_val, y_val, config):
    # 创建模型保存目录
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'lstm_model')  # 移除.h5后缀，因为这将是一个目录
    
    # 检查是否存在已保存的模型
    if os.path.exists(model_path):
        print(f"\n发现已保存的模型，正在加载: {model_path}")
        # 使用 custom_objects 来正确加载损失函数
        simple_model = tf.keras.models.load_model(model_path)
        return simple_model, None  # 返回None作为history，因为是加载的模型
    
    print("\n未找到已保存的模型，开始训练新模型...")
    simple_model = create_simple_model((config["input_length"], 1))
    # 确保模型在训练前已经编译
    simple_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )
    
    simple_history = train_models(
        simple_model, X_train, y_train, X_val, y_val, epochs=config["epochs"]
    )
    
    # 保存模型
    simple_model.save(model_path, save_format='tf')
    print(f"\n模型已保存到: {model_path}")
    
    return simple_model, simple_history

async def main():
    setup_logger(clear_log=True)
    args = load_args()

    config = load_config(args.config)

    client = IBClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id
    )

    data = await client.getHistoricalData()

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(data, config)

    simple_model, simple_history = create_and_train_model(
        X_train, y_train, X_val, y_val, config
    )

    # 只在有训练历史时进行评估（即新训练的模型）
    if simple_history is not None:
        evaluate_models(
            simple_model,
            simple_history,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

    # 使用最后一段数据进行预测
    # features = ['close', 'volume']
    # last_sequence = data[features].iloc[-config["input_length"]:].values.reshape(1, config["input_length"], 2)

    # 从每个数据集中选择一个样本进行预测
    train_sample = X_train[0:1]  # 取第一个样本，保持(1, sequence_length, features)的形状
    val_sample = X_val[0:1]
    test_sample = X_test[0:1]

    # 进行预测
    train_pred = simple_model.predict(train_sample)[0][0]  # 获取预测值
    val_pred = simple_model.predict(val_sample)[0][0]
    test_pred = simple_model.predict(test_sample)[0][0]

    # 打印预测结果
    print("\n预测结果示例:")
    print(f"训练集 - 预测值: {train_pred:.4f}, 真实值: {y_train[0]:.4f}")
    print(f"验证集 - 预测值: {val_pred:.4f}, 真实值: {y_val[0]:.4f}")
    print(f"测试集 - 预测值: {test_pred:.4f}, 真实值: {y_test[0]:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
