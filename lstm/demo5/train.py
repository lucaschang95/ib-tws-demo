import tensorflow as tf

def get_callbacks():
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
    callbacks = get_callbacks()
    
    print("Training simple model...")
    simple_history = simple_model.fit(
        X, y,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=False,
        verbose=1  # Show progress bar and metrics
    )
    
    # 创建保存模型的目录
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    # 保存模型（SavedModel格式）
    simple_model.save('saved_models/lstm_model')
    print("模型已保存到 saved_models/lstm_model 目录")
    
    # 保存模型（H5格式）
    simple_model.save('saved_models/lstm_model.h5')
    print("模型已保存到 saved_models/lstm_model.h5 文件")
    
    return simple_history


# # 加载SavedModel格式的模型
# loaded_model = tf.keras.models.load_model('saved_models/lstm_model')

# # 或者加载H5格式的模型
# loaded_model = tf.keras.models.load_model('saved_models/lstm_model.h5')