import os

import pandas as pd
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

    # client = IBClient(
    #     host=args.host,
    #     port=args.port,
    #     client_id=args.client_id
    # )

    # data = await client.getHistoricalData()
    data = pd.read_csv('test-data/aapl_bars.csv')
    
    train_gen, val_gen, test_gen = preprocess(data, config)
    
    # 释放原始数据内存
    del data
    
    simple_model = create_model_with_config(config)
    
    # Create trainer instance and train the model
    trainer = LSTMTrainer(config)
    simple_history = trainer.train(
        model=simple_model,
        train_data=train_gen,
        val_data=val_gen
    )

    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'lstm_model.keras')
    simple_model.save(model_path)
    print(f"\n模型已保存到: {model_path}")

    # 只在有训练历史时进行评估（即新训练的模型）
    if simple_history is not None:
        evaluate_models(
            simple_model,
            simple_history,
            train_gen=train_gen,
            val_gen=val_gen,
            test_gen=test_gen
        )

    # 从每个生成器中获取一个批次进行预测示例
    train_batch = next(iter(train_gen))
    val_batch = next(iter(val_gen))
    test_batch = next(iter(test_gen))

    # 进行预测
    train_pred = simple_model.predict(train_batch[0], verbose=0)[0, -1, 0]
    val_pred = simple_model.predict(val_batch[0], verbose=0)[0, -1, 0]
    test_pred = simple_model.predict(test_batch[0], verbose=0)[0, -1, 0]

    # 还原预测值和真实值（仅还原收盘价）
    mean, std = config['close_mean'], config['close_std']
    
    # 打印预测结果
    print("\n预测结果示例:")
    print(f"训练集 - 预测值: {train_pred * std + mean:.2f}, 真实值: {train_batch[1][0] * std + mean:.2f}")
    print(f"验证集 - 预测值: {val_pred * std + mean:.2f}, 真实值: {val_batch[1][0] * std + mean:.2f}")
    print(f"测试集 - 预测值: {test_pred * std + mean:.2f}, 真实值: {test_batch[1][0] * std + mean:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
