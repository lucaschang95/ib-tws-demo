import numpy as np

def prepare_test_input(start=50, length=99):
    """Prepare test input data"""
    test_input = np.array(range(start, start + length))
    return test_input.reshape((1, length, 1))

def evaluate_models(simple_model, X, y, test_input):
    """Evaluate and compare models"""
    simple_prediction = simple_model.predict(test_input, verbose=0)
    
    print("\nModel Comparison:")
    print(f'Simple model predicted value: {simple_prediction[0][0]}')
    
    simple_mse = simple_model.evaluate(X, y, verbose=0)
    
    print(f'\nSimple model MSE: {simple_mse}')
