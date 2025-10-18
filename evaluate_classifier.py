# evaluate_classifier.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

MODEL_PATH = 'models/waste_classifier_mobilenet.h5'
TEST_DIR = 'dataset/test'

# Load model
model = load_model(MODEL_PATH)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=(224,224), batch_size=32, shuffle=False)

# Predict
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
labels = list(test_gen.class_indices.keys())

# Accuracy
acc = np.mean(y_true == y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
os.makedirs('results', exist_ok=True)
plt.savefig('results/confusion_matrix.png')
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('results/classification_report.csv')
print(df)
