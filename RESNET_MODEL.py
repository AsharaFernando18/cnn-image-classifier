import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import mixed_precision
import numpy as np

# Enable XLA
tf.config.optimizer.set_jit(True)

# Enable mixed precision
mixed_precision.set_global_policy("mixed_float16")

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Simplified preprocessing for 60-70% accuracy (faster training)
def preprocess_train(image, label):
    image = tf.image.resize(image, [48, 48])  # Smaller size for faster training
    image = tf.cast(image, tf.float32) / 255.0
    # Reduced data augmentation for speed
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label

def preprocess_test(image, label):
    image = tf.image.resize(image, [48, 48])  # Match training size
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64  # Optimal batch size for accuracy

train_ds = tf.data.Dataset.from_tensor_slices((x_train, to_categorical(y_train, 10)))
train_ds = train_ds.shuffle(10000).map(preprocess_train, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).repeat().prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, to_categorical(y_test, 10)))
test_ds = test_ds.map(preprocess_test, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Learning rate schedule optimized for 60-70% accuracy
def lr_schedule(epoch):
    if epoch < 5:
        return 5e-4
    elif epoch < 10:
        return 1e-4
    else:
        return 5e-5

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),  # Increased patience for better training
    ReduceLROnPlateau(patience=3, factor=0.2, verbose=1, min_lr=1e-8),  # More aggressive LR reduction
    ModelCheckpoint("best_resnet50_cifar10.h5", save_best_only=True, save_weights_only=False),
    lr_scheduler
]

# Build model optimized for 60-70% accuracy
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Simplified architecture
x = tf.keras.layers.Dropout(0.4)(x)
predictions = Dense(10, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 🔒 Phase 1: Train with frozen base
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

print("🧊 Phase 1: Training with frozen ResNet50...")
model.fit(
    train_ds,
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=8,  # Reduced epochs for faster training
    validation_data=test_ds,
    validation_steps=len(x_test) // BATCH_SIZE,
    callbacks=callbacks
)

# 🔓 Phase 2: Fast fine-tuning for 60-70% accuracy
for layer in base_model.layers[-15:]:  # Unfreeze fewer layers for speed
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),  # Smaller learning rate for fine-tuning
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

print("🔥 Phase 2: Fast fine-tuning for 60-70% accuracy...")
model.fit(
    train_ds,
    steps_per_epoch=(len(x_train) // BATCH_SIZE) // 2,  # Half steps for speed
    epochs=8,  # Reduced epochs for faster Phase 2
    validation_data=test_ds,
    validation_steps=(len(x_test) // BATCH_SIZE) // 2,  # Half validation steps
    callbacks=callbacks
)

# Save model
model.save("resnet50_cifar10_model_finetuned.h5")
print("✅ Model saved as resnet50_cifar10_model_finetuned.h5")