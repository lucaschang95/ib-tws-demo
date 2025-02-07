import tensorflow as tf

class PrecisionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        metrics_str = []
        for metric, value in logs.items():
            metrics_str.append(f"{metric}: {value:.6f}")
        print(f"\nEpoch {epoch + 1}: {' - '.join(metrics_str)}")

def get_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        min_delta=1e-4,
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-6
    )

    return [early_stopping, lr_scheduler]

def train_models(simple_model, X_train, y_train, X_val, y_val, epochs=200):
    callbacks = get_callbacks()
    
    print("Training simple model...")
    simple_history = simple_model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val), # 使用单独的验证集
        callbacks=callbacks,
        shuffle=False,
        verbose=1,  # Show progress bar and metrics
    )

    return simple_history