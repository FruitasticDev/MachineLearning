import numpy as np
import tensorflow as tf
import io, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # Mengizinkan permintaan dari semua domain

# Menggunakan model TensorFlow
model_keras = tf.keras.models.load_model('fruit_classifier_final.keras')  # Sesuaikan path model Anda
model_h5 = tf.keras.models.load_model('fruit_classifier_final.h5')  # Sesuaikan path model Anda

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # # Mengambil gambar dari request
        # img_file = request.files['image']
        # img = Image.open(io.BytesIO(img_file.read()))
        # img = img.resize((224, 224))  # Ukuran gambar yang diinginkan
        # img_array = np.array(img) / 255.0  # Normalisasi gambar
        # img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension

        # Get image URL from request
        data = request.json
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400

        # Download image from URL
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = img.resize((224, 224))  # Ukuran gambar yang diinginkan
        img_array = np.array(img) / 255.0  # Normalisasi gambar
        img_array = np.expand_dims(img_array, axis=0)  # Menambahkan batch dimension

        # Prediksi menggunakan model
        predictions = model_keras.predict(img_array)
        # predictions = model_h5.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])


        return jsonify({
            'predictions': predictions.tolist(),
            'maxPredictionIndex': int(predicted_class),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)  # Server Flask berjalan di port 3000
