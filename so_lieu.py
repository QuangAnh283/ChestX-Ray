import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

MODEL_PATH = "chest_xray_cnn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

test_dir = "chest_xray/chest_xray"  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")

y_pred = (model.predict(test_generator) > 0.5).astype("int32")
y_true = test_generator.classes

labels = list(test_generator.class_indices.keys())

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img_path = test_generator.filepaths[i]
    img = plt.imread(img_path)
    plt.imshow(img, cmap='gray')
    pred_label = labels[y_pred[i][0]]
    true_label = labels[y_true[i]]
    color = "green" if pred_label == true_label else "red"
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
print("\n🔢 Confusion Matrix:")
print(cm)
