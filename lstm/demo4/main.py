import pandas as pd
from data_generator import prepare_data, generate_sine_wave, create_sequences, normalize_data
from models import create_simple_model
from train import train_models
from evaluate import prepare_test_input, evaluate_models

def main():
    print("Starting LSTM demo...")

    input_length = 100
    amplitude = 300
    n_points = 3000
    
    # 生成带噪声的正弦波数据
    time, wave_data = generate_sine_wave(n_points=n_points, amplitude=amplitude)

    wave_data = normalize_data(wave_data)
    print("wave_data normalized, first 5 time points:", wave_data[:5])

    # 创建序列数据
    X, y = create_sequences(wave_data, sequence_length=input_length)
    # 准备LSTM输入格式
    X = prepare_data(X)

        # Evaluate models
    test_input = prepare_test_input(input_length, amplitude)

    test_input = normalize_data(test_input.reshape(-1)).reshape(1, input_length, 1)

    simple_model = create_simple_model((input_length, 1))
    
    # Train models
    simple_history = train_models(X, y, simple_model, epochs=5000)
    
    # Evaluate models
    test_input = prepare_test_input(input_length, amplitude)

    test_input = normalize_data(test_input.reshape(-1)).reshape(1, input_length, 1)

    evaluate_models(simple_model, X, y, test_input)

if __name__ == "__main__":
    main()
