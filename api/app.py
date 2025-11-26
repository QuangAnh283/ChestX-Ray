from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'mobilenetv2_finetuned_optimized.h5')
MODEL_PATH = os.path.abspath(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('index.html', error="Vui lòng chọn ảnh trước khi dự đoán.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)


        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        result = CLASS_NAMES[predicted_class]

        return render_template(
            'index.html',
            file_path=file_path,
            result=result,
            confidence=round(confidence * 100, 2)
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
