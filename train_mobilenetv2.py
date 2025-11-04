from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Print start message
print("Starting the training process...")
print("Setting up data generators...")

# Image generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("Loading training data...")
try:
    train_data = train_datagen.flow_from_directory(
        'asl_alphabet_train/asl_alphabet_train',
        target_size=(224, 224),
        batch_size=16,  # Reduced batch size
        subset='training',
        shuffle=True,
        class_mode='categorical'
    )
    print(f"Found {train_data.samples} training images in {train_data.num_classes} classes")
except Exception as e:
    print(f"Error loading training data: {str(e)}")
    exit(1)

print("Loading validation data...")
try:
    val_data = train_datagen.flow_from_directory(
        'asl_alphabet_train/asl_alphabet_train',
        target_size=(224, 224),
        batch_size=32,
        subset='validation',
        shuffle=True,
        class_mode='categorical'
    )
    print(f"Found {val_data.samples} validation images")
except Exception as e:
    print(f"Error loading validation data: {str(e)}")
    exit(1)

print("\nInitializing MobileNetV2 model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])


# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train with callbacks
print("\nStarting model training...")
try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=[reduce_lr, checkpoint, early_stopping],
        verbose=1
    )
except Exception as e:
    print(f"\nError during training: {str(e)}")
    exit(1)

# Plot training history

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ Training history plot saved as 'training_history.png'")

# Save final model
model.save('asl_mobilenetv2.h5')
print("✅ Model saved as asl_mobilenetv2.h5")
