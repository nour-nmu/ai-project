"""
Improved MobileNetV2 Training Script for ASL Translator
-------------------------------------------------------
Enhancements:
- Advanced data augmentation
- Two-phase training (feature extraction + fine-tuning)
- Learning rate scheduling and gradient clipping
- Regularization layers for better generalization
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
)
import matplotlib.pyplot as plt
import tensorflow as tf

print("🚀 Starting the improved MobileNetV2 training...")

# =======================
# Data Preprocessing
# =======================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    'asl_alphabet_train/asl_alphabet_train',
    target_size=(224, 224),
    batch_size=32,
    subset='training',
    shuffle=True,
    class_mode='categorical'
)

val_data = train_datagen.flow_from_directory(
    'asl_alphabet_train/asl_alphabet_train',
    target_size=(224, 224),
    batch_size=32,
    subset='validation',
    shuffle=False,
    class_mode='categorical'
)

# =======================
# Model Setup
# =======================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Phase 1: Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# =======================
# Compilation
# =======================
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =======================
# Callbacks
# =======================
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    CSVLogger('training_log.csv', append=False)
]

# =======================
# Phase 1: Feature Extraction
# =======================
print("\n🧠 Phase 1: Training top layers (frozen base model)...")
history_1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# =======================
# Phase 2: Fine-Tuning
# =======================
print("\n🔧 Phase 2: Fine-tuning deeper layers...")

for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# =======================
# Combine History
# =======================
history = {
    'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
    'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
    'loss': history_1.history['loss'] + history_2.history['loss'],
    'val_loss': history_1.history['val_loss'] + history_2.history['val_loss']
}

# =======================
# Plot Results
# =======================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('improved_training_history.png')
print("✅ Saved improved training plot as 'improved_training_history.png'")

# =======================
# Save Final Model
# =======================
model.save('asl_mobilenetv2_improved.h5')
print("✅ Saved fine-tuned model as 'asl_mobilenetv2_improved.h5'")
