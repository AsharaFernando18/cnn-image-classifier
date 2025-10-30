#!/usr/bin/env python3
"""
Quick Model Retraining Script - Fast Version
Simplified training for quick model updates
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def quick_retrain(epochs=10, save_model=True):
    """
    Quick retraining function with fewer epochs for testing
    """
    print("🚀 QUICK RETRAINING MODE")
    print("=" * 50)
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # Preprocessing
    print("Preprocessing data...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    print("✅ Preprocessing complete!")
    
    # Build enhanced model
    print("Building enhanced CNN model...")
    model = Sequential([
        # First block
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third block (FIXED: Removed problematic pooling)
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        # No MaxPooling2D here to avoid dimension issues
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Training configuration
    batch_size = 32
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    print("\n🎯 EVALUATION RESULTS:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test Loss: {loss:.4f}")
    
    # Save model if requested
    if save_model:
        model_path = 'cifar10_cnn_model_retrained.h5'
        model.save(model_path)
        print(f"✅ Model saved as: {model_path}")
    
    # Quick visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quick_retrain_history.png', dpi=300, bbox_inches='tight')
    print("✅ Training history saved as: quick_retrain_history.png")
    plt.close()
    
    print("\n🎉 RETRAINING COMPLETE!")
    return model, history

def load_existing_and_continue_training(epochs=5):
    """
    Load existing model and continue training
    """
    print("📦 CONTINUE TRAINING MODE")
    print("=" * 50)
    
    try:
        # Load existing model
        model = tf.keras.models.load_model('cifar10_cnn_model.h5')
        print("✅ Existing model loaded successfully!")
        
        # Load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        print(f"Continuing training for {epochs} additional epochs...")
        
        # Continue training
        history = model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Save updated model
        model.save('cifar10_cnn_model_continued.h5')
        print("✅ Updated model saved as: cifar10_cnn_model_continued.h5")
        
        return model, history
        
    except FileNotFoundError:
        print("❌ No existing model found. Use quick_retrain() instead.")
        return None, None

def train_with_different_parameters(epochs=15, learning_rate=0.0005, batch_size=64):
    """
    Retrain with different hyperparameters
    """
    print("⚙️ CUSTOM PARAMETER TRAINING")
    print("=" * 50)
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Build model with custom optimizer
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        # No MaxPooling2D here to avoid dimension issues
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    # Custom optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    # Save
    model.save(f'cifar10_cnn_model_lr{learning_rate}_bs{batch_size}.h5')
    print(f"✅ Model saved with custom parameters")
    
    return model, history

if __name__ == "__main__":
    print("🔄 CNN Model Retraining Options")
    print("=" * 50)
    print("1. Quick retrain (10 epochs) - Recommended for testing")
    print("2. Continue training existing model")
    print("3. Custom parameter training")
    
    choice = input("\nEnter your choice (1/2/3): ")
    
    if choice == "1":
        model, history = quick_retrain(epochs=10)
    elif choice == "2":
        model, history = load_existing_and_continue_training(epochs=5)
    elif choice == "3":
        epochs = int(input("Enter epochs (default 15): ") or 15)
        lr = float(input("Enter learning rate (default 0.0005): ") or 0.0005)
        bs = int(input("Enter batch size (default 64): ") or 64)
        model, history = train_with_different_parameters(epochs, lr, bs)
    else:
        print("Quick retraining with default settings...")
        model, history = quick_retrain()
