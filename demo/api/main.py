# Face Antispoofing Demo and Testing
# Muhammad Adisatriyo Pratama - June 2021

from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__, template_folder='template')
model = load_model("assets/MobileNetV2.h5")

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route('/')
def upload_file():
   return render_template('upload.html')

@app.route("/image", methods=['GET', 'POST'])
def process_image():
    class_names = ["real", "spoof"]
    img_height = img_width = 224
    file = request.files['file']
    # Read the image via file.stream
    img = Image.open(file.stream)
    img = img.resize((224, 224), Image.ANTIALIAS)
    # resized_img = keras.preprocessing.image.load_img(img, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_preprocessed = preprocess_input(img_array)

    prediction = model.predict(img_preprocessed)
    score = tf.nn.softmax(prediction[0])
    
    
    return jsonify({'msg': 'success', 'size': [img.width, img.height], 'prediction' : str(class_names[np.argmax(score)]), 'score': 100 * np.max(score)})



if __name__ == "__main__":
    app.run(debug=False)