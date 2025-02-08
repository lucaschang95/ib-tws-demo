import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any

class BaseTrainer:
    """基础训练器类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history = None
        
    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """获取回调函数列表"""
        raise NotImplementedError
        
    def train(self, model: tf.keras.Model, 
              train_data: Tuple[tf.Tensor, tf.Tensor],
              val_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.keras.callbacks.History:
        """训练模型的抽象方法"""
        raise NotImplementedError

class LSTMTrainer(BaseTrainer):
    """LSTM模型训练器"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 200)
        self.shuffle = config.get('shuffle', False)
        self.verbose = config.get('verbose', 1)

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """配置训练回调函数"""
        callbacks = []
        
        # 精确度回调
        class PrecisionCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                metrics_str = []
                for metric, value in logs.items():
                    metrics_str.append(f"{metric}: {value:.6f}")
                print(f"\nEpoch {epoch + 1}: {' - '.join(metrics_str)}")
        
        callbacks.append(PrecisionCallback())
        
        # 早停
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 50),
            min_delta=self.config.get('early_stopping_min_delta', 1e-4),
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # 学习率调度器
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.get('lr_factor', 0.5),
            patience=self.config.get('lr_patience', 20),
            min_lr=self.config.get('min_lr', 1e-6)
        )
        callbacks.append(lr_scheduler)
        
        return callbacks

    def train(self, model: tf.keras.Model,
             train_data: Tuple[tf.Tensor, tf.Tensor],
             val_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.keras.callbacks.History:
        """训练模型"""
        X_train, y_train = train_data
        callbacks = self._get_callbacks()
        
        self.history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            shuffle=self.shuffle,
            verbose=self.verbose,
        )
        
        return self.history
    
    def get_training_history(self) -> Optional[tf.keras.callbacks.History]:
        """获取训练历史"""
        return self.history 