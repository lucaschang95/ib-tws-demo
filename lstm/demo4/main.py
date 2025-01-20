import numpy as np
import tensorflow as tf
from data_generator import prepare_data, generate_sine_wave, create_sequences
from models import create_simple_model
from train import train_models
from evaluate import prepare_test_input, evaluate_models

def main():
    print("Starting LSTM demo...")
    
    # Define input sequence length
    input_length = 9
    
    # 生成带噪声的正弦波数据
    wave_data = generate_sine_wave(n_points=300, amplitude=10)
    print(wave_data)
    exit();

    # 创建序列数据
    X, y = create_sequences(wave_data, sequence_length=input_length)
    
    # 准备LSTM输入格式
    X = prepare_data(X)
    
    # Create models
    simple_model = create_simple_model((input_length, 1))
    
    # Train models
    simple_history = train_models(X, y, simple_model, epochs=5000)
    
    # Evaluate models
    test_input = prepare_test_input(400, input_length)
    evaluate_models(simple_model, X, y, test_input)

if __name__ == "__main__":
    main()
