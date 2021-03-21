#  Copyright (c) 2021
#  This script is prepared by TechyHans (https://techyhans.com)
#  Anything please contact him at hanshengliang@outlook.com for more details.

from flask import Flask, request,jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask_cors import CORS

model = keras.models.load_model('model.h5')
model2 = keras.models.load_model('model-strawberry-v1.h5')

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/')
def hello():
    return "Hello World!"

@app.route('/login', methods=["POST"])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == "admin" and password == "admin":
        return jsonify({"message": "success"})
    else:
        return jsonify({"message": "fail"})

@app.route('/predict-strawberry', methods=["POST"])
def predict_strawberry():
    try:
        ori_file = request.files['uploaded_image']
        ori_file.save("temp2.jpg")

        image_path = "temp2.jpg"
        img_height = 224
        img_width = 224

        class_names = ['Leaf Scorch', 'Healthy']

        img = keras.preprocessing.image.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model2.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        pred_class = class_names[np.argmax(score)]
        pred_score = round(100 * np.max(score), 2)

        details={
            "predictions": {
                "type": "strawberry",
                "probability": pred_score,
                "tagName": pred_class,
            }
        }
        return jsonify(details)
    except:
        return jsonify({"Error":"System Error"})

@app.route('/predict-tomato', methods=["POST"])
def predict_tomato():
    try:
        ori_file = request.files['uploaded_image']
        ori_file.save("temp.jpg")

        image_path = "temp.jpg"
        img_height = 224
        img_width = 224

        class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot',
                    'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus',
                    'Two-spotted Spider Mite', 'Healthy']

        img = keras.preprocessing.image.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        pred_class = class_names[np.argmax(score)]
        pred_score = round(100 * np.max(score), 2)

        details={
            "predictions": {
                "type": "tomato",
                "probability": pred_score,
                "tagName": pred_class,
            }
        }
        return jsonify(details)
    except:
        return jsonify({"Error":"System Error"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)