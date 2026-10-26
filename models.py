import tensorflow as tf
from tensorflow.keras import layers, models

def build_1d_cnn(input_shape=(540, 1), num_classes=6):
    """
    Constructs the custom 1D CNN architecture for ECG diagnosis.
    Architecture: Conv1D-ReLU-MaxPool -> Conv1D-ReLU-MaxPool -> Dropout -> Flatten -> Dense-ReLU-Dropout -> Dense-Softmax
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        
        # Block 2
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        
        # Classifier Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model