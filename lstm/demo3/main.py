import numpy as np
import tensorflow as tf
from data_generator import generate_sequence_data, prepare_data
from models import create_simple_model
from train import train_models
from evaluate import prepare_test_input, evaluate_models

def main():
    print("Starting LSTM demo...")
    
    # Define input sequence length
    input_length = 9
    
    # Generate and prepare data
    data = generate_sequence_data(1000, input_length, 3000)
    X, y = prepare_data(data)
    
    # Create models
    simple_model = create_simple_model((input_length, 1))
    
    # Train models
    simple_history = train_models(X, y, simple_model, epochs=5000)
    
    # Evaluate models
    test_input = prepare_test_input(400, input_length)
    evaluate_models(simple_model, X, y, test_input)

if __name__ == "__main__":
    main()
