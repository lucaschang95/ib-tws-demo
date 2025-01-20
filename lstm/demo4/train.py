import tensorflow as tf

def get_callbacks():
    """Get training callbacks"""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    return [early_stopping, lr_scheduler]

def train_models(X, y, simple_model, epochs=200, validation_split=0.2):
    """Train both models with early stopping and learning rate scheduling
    
    Args:
        X: Input features
        y: Target values
        simple_model: The model to train
        epochs: Maximum number of epochs
        validation_split: Fraction of data to use for validation
    """
    callbacks = get_callbacks()
    
    print("\nTraining simple model...")
    simple_history = simple_model.fit(
        X, y,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1  # Show progress bar and metrics
    )
    
    return simple_history
