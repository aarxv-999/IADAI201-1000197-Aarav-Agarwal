# eval_waste_model.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os
import itertools

# SETTINGS
MODEL_PATH = "waste_classifier.h5"
TEST_DIR = "dataset/test"   # point this to your test folder
IMG_SIZE = (224, 224)
BATCH = 32
CLASS_NAMES = ['Biodegradable', 'Recyclable', 'Hazardous']  # ensure ordering matches generator

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# Test generator (no augmentation, only rescale)
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

# Predict
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# Report
print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(test_gen.class_indices.keys()),
            yticklabels=list(test_gen.class_indices.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (counts)')
plt.savefig('confusion_matrix_counts.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(7,6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=list(test_gen.class_indices.keys()),
            yticklabels=list(test_gen.class_indices.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (normalized)')
plt.savefig('confusion_matrix_norm.png', bbox_inches='tight')
plt.close()

# Overall accuracy
acc = np.sum(y_pred == y_true) / len(y_true)
print(f"\nTest accuracy: {acc*100:.2f}%")

# Save raw preds and y_true if needed
np.save("y_true.npy", y_true)
np.save("y_pred.npy", y_pred)
np.save("pred_probs.npy", preds)

print("✅ Evaluation complete. Confusion matrices saved.")
