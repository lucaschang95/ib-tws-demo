import numpy as np
import tensorflow as tf
from data_generator import generate_sequence_data, prepare_data
from models import create_simple_model
from train import train_models
from evaluate import prepare_test_input, evaluate_models

def main():
    print("Starting LSTM demo...")
    
    # Generate and prepare data
    data = generate_sequence_data(100, 9)
    X, y = prepare_data(data)
    
    # Create models
    simple_model = create_simple_model()
    # complex_model = create_complex_model()
    
    # Train models
    simple_history = train_models(X, y, simple_model)
    
    # Evaluate models
    test_input = prepare_test_input(120, 9)
    evaluate_models(simple_model, X, y, test_input)

if __name__ == "__main__":
    main()
