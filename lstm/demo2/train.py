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

def train_models(X, y, simple_model):
    """Train both models"""
    print("\nTraining simple model...")
    simple_history = simple_model.fit(X, y, epochs=200, verbose=0)
    
    print("\nTraining complex model...")
    # complex_history = complex_model.fit(
    #     X, y,
    #     epochs=500,
    #     batch_size=32,
    #     validation_split=0.2,
    #     callbacks=get_callbacks(),
    #     verbose=0
    # )
    
    return simple_history
