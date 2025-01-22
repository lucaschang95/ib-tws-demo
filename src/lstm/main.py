import json
import logging
import os
from data_generator import (
    prepare_data,
    generate_sine_wave,
    create_sequences,
    normalize_data,
    split_data,
)
from models import create_simple_model
from train import train_models
from evaluate import evaluate_models


script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "app.log")
# 清空日志文件
with open(log_file, "w") as f:
    f.write("")

# 配置日志记录
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def prepare_data_and_sequences(config):
    """准备数据和序列。"""
    time, wave_data = generate_sine_wave(
        n_points=config["n_points"], amplitude=config["amplitude"]
    )
    wave_data = normalize_data(wave_data)

    # 划分数据集
    train_data, val_data, test_data = split_data(wave_data)

    # 训练集
    X_train, y_train = create_sequences(
        train_data, sequence_length=config["input_length"]
    )
    X_train = prepare_data(X_train)

    # 验证集
    X_val, y_val = create_sequences(val_data, sequence_length=config["input_length"])
    X_val = prepare_data(X_val)

    # 测试集
    X_test, y_test = create_sequences(test_data, sequence_length=config["input_length"])
    X_test = prepare_data(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_and_train_model(X_train, y_train, X_val, y_val, config):
    """创建并训练模型。"""
    simple_model = create_simple_model((config["input_length"], 1))
    simple_history = train_models(
        simple_model, X_train, y_train, X_val, y_val, epochs=config["epochs"]
    )
    return simple_model, simple_history


def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    logging.info("Starting LSTM demo...")

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_and_sequences(config)
    simple_model, simple_history = create_and_train_model(
        X_train, y_train, X_val, y_val, config
    )

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


if __name__ == "__main__":
    main()
