from flask import Flask, request, jsonify
import random
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io

import tempfile
from keras import models
import boto3

# Creating the low level functional client
client = boto3.client(
    's3',
    aws_access_key_id = 'AKIA5H64GCH7LG5OMBOJ',
    aws_secret_access_key = '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',
    region_name = 'us-west-2'
)


# Create the S3 object
response_data = client.get_object(
    Bucket = 'camera-ml',
    Key = 'TillingModel.h5'
)

model_name='model.h5'
response_data=response_data['Body']
response_data=response_data.read()
#save byte file to temp storage
with tempfile.TemporaryDirectory() as tempdir:
    with open(f"{tempdir}/{model_name}", 'wb') as my_data_file:
        my_data_file.write(response_data)
        #load byte file from temp storage into variable
        gotten_model=models.load_model(f"{tempdir}/{model_name}")



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
    yhat = gotten_model.predict(test_image)
    p=np.argmax(yhat)+1
    print(p)
    # Assuming here that the file is valid and you've done whatever processing you need
    return jsonify({'message': 'Image successfully uploaded', 'predictedTillage': int(p)})

if __name__ == '__main__':
    app.run(debug=True)
