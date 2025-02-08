from models.lstm import create_model_with_config, evaluate_models
from utils.args import load_args
from utils.logger import setup_logger
from utils.config import load_config
from ib.client import IBClient
import asyncio
from utils.preprocess import preprocess
from models.lstm import create_model_with_config
from models.trainer import LSTMTrainer

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

    simple_model = create_model_with_config(config)
    
    # Create trainer instance and train the model
    trainer = LSTMTrainer(config)
    simple_history = trainer.train(
        model=simple_model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val)
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

    # 从每个数据集中选择一个样本进行预测
    train_sample = X_train[0:1]  # 取第一个样本，保持(1, sequence_length, features)的形状
    val_sample = X_val[0:1]
    test_sample = X_test[0:1]

    # 进行预测
    train_pred = simple_model.predict(train_sample, verbose=0)[0, -1, 0]  # 获取最后一个时间步的预测值
    val_pred = simple_model.predict(val_sample, verbose=0)[0, -1, 0]
    test_pred = simple_model.predict(test_sample, verbose=0)[0, -1, 0]

    # 还原预测值和真实值（仅还原收盘价）
    mean, std = config['close_mean'], config['close_std']
    
    # 打印预测结果
    print("\n预测结果示例:")
    print(f"训练集 - 预测值: {train_pred * std + mean:.2f}, 真实值: {y_train[0].item() * std + mean:.2f}")
    print(f"验证集 - 预测值: {val_pred * std + mean:.2f}, 真实值: {y_val[0].item() * std + mean:.2f}")
    print(f"测试集 - 预测值: {test_pred * std + mean:.2f}, 真实值: {y_test[0].item() * std + mean:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
