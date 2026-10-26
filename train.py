import numpy as np
from src.models import build_1d_cnn
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    """Trains the 1D CNN model with checkpointing and early stopping."""
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train.shape[1]
    
    model = build_1d_cnn(input_shape=input_shape, num_classes=num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_ecg_model.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history