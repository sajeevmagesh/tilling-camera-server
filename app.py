from flask import Flask, request, jsonify
import random
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io

import logging

try:
    loaded_model = tf.keras.models.load_model('TillingModel.h5')
except Exception as e:
    logging.exception("Failed to load the TensorFlow model")
    raise  # Optionally re-raise the exception to halt the application



app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400
    img_bytes = io.BytesIO(file.read())
    test_image = image.load_img(img_bytes, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)
    yhat = loaded_model.predict(test_image)
    p=np.argmax(yhat)+1
    print(p)
    # Assuming here that the file is valid and you've done whatever processing you need
    return jsonify({'message': 'Image successfully uploaded', 'predictedTillage': int(p)})

if __name__ == '__main__':
    app.run(debug=True)
