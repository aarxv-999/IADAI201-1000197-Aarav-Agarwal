# train_classifier.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Paths
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'
MODEL_PATH = 'models/waste_classifier_mobilenet.h5'

# Image parameters
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 20

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True
).flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH)

# Base model (MobileNetV2)
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

# Freeze base model initially
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Phase 1 training
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Fine-tune last few layers
for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model
os.makedirs('models', exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
