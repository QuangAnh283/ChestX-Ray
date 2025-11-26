import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

model = tf.keras.models.load_model("models/mobilenetv2_finetuned_optimized.h5")

class_labels = ["NORMAL", "PNEUMONIA"]  

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    print(f"🩺 Kết quả dự đoán: {class_labels[predicted_class]} ({confidence * 100:.2f}%)")

predict_image("chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg")
