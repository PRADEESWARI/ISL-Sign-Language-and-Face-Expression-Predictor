import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- PATHS ---
MODEL_PATH = "face_cnn_model.h5"
TEST_DIR = "fer_dataset/test"

# --- PARAMETERS ---
IMG_SIZE = (48, 48)   # use same as training
BATCH_SIZE = 64

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ FER model loaded.")

# --- LOAD TEST DATA ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- PREDICT ---
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- ACCURACY ---
acc = accuracy_score(y_true, y_pred)
print(f"📊 Test Accuracy: {acc*100:.2f}%")

# --- CLASSIFICATION REPORT ---
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('FER Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
