import numpy as np
import tensorflow as tf
import io, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # Mengizinkan permintaan dari semua domain

# Menggunakan model TensorFlow
model_keras = tf.   keras.models.load_model('fruit_classifier_final.keras')  # Sesuaikan path model Anda
model_h5 = tf.keras.models.load_model('fruit_classifier_final.h5')  # Sesuaikan path model Anda

index_name = ['AvocadoQ_Fresh', 'AvocadoQ_Mild', 'AvocadoQ_Rotten', 'BananaDB_Fresh', 'BananaDB_Mild', 'BananaDB_Rotten', 'CucumberQ_Fresh', 'CucumberQ_Mild', 'CucumberQ_Rotten', 'GrapefruitQ_Fresh', 'GrapefruitQ_Mild', 'GrapefruitQ_Rotten', 'KakiQ_Fresh', 'KakiQ_Mild', 'KakiQ_Rotten', 'PapayaQ_Fresh', 'PapayaQ_Mild', 'PapayaQ_Rotten', 'PeachQ_Fresh', 'PeachQ_Mild', 'PeachQ_Rotten', 'tomatoQ_Fresh', 'tomatoQ_Mild', 'tomatoQ_Rotten']


def load_and_preprocess_image(image, img_size=(224, 224)):
    """
    Preprocess an image for model prediction in Flask.

    Args:
        image (PIL.Image): Input image
        img_size (tuple): Target image size

    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Resize the image
    img = image.resize(img_size)

    # Convert to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)

    # Apply ResNet preprocessing
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image URL from request
        data = request.json
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400

        # Download image from URL
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))

        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Preprocess the image using the same method as in the notebook
        img_array = load_and_preprocess_image(img)

        # Predict using the model
        predictions = model_keras.predict(img_array)

        # Get the predicted class index
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Get the class label
        predicted_label = index_name[predicted_class]

        # Get the confidence score
        confidence = float(predictions[0][predicted_class])

        return jsonify({
            'predictions': predictions.tolist(),
            'maxPredictionIndex': int(predicted_class),
            'predictedLabel': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)  # Server Flask berjalan di port 3000
