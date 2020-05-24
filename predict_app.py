import base64
import numpy as np
import cv2
import io
import os
from PIL import Image
import keras
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from scipy import misc
import skimage
from skimage.transform import resize
from flask import request
from flask import jsonify
from flask import Flask

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)

def get_model():
    global  model
    model = load_model('pneumonia_identifier.h5')
    print(" * Model Loaded!")

def get_data(Dir):
    X = []
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['NORMAL']:
                label = 0
            elif nextDir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
                
            temp = Dir + nextDir
                
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = skimage.transform.resize(img, (150, 150, 3))
                    #img_file = scipy.misc.imresize(arr=img_file, size=(299, 299, 3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)
                    
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

def preporcess_image(image, target_size):
    X = []
    img = np.asarray(image)
    if img is not None:
        img = skimage.transform.resize(img, (target_size))   #using scikit-image, transform i.e resize the image
        img = np.asarray(img)
        X.append(img)

    X = np.asarray(X) 
    # X = img.reshape(1,3,150,150)
    # if image.mode != 'RGB':
    #     image = image.convert("RGB")
    # image = image.resize(target_size)
    # image = np.asarray(image)
    # image = np.reshape(3,150,150)

    return X

print(" * Loading Keras Model....")
get_model()
model._make_predict_function()
graph = tf.get_default_graph()

@app.route("/predict", methods=['POST'])
def predict():
    a,b = get_data('./test/')
    # message = request.get_json(force=True)
    # encoded = message['image']
    # decoded = base64.b64decode(encoded)
    # image = Image.open(io.BytesIO(decoded))
    # processed_image = preporcess_image(image, target_size=(150,150,3))
    # print(processed_image.shape)
    # processed_image = processed_image.reshape(1,3,150,150)
    a = a.reshape(1,3,150,150)

    global graph
    with graph.as_default():
        prediction = model.predict(a).tolist()       
    # prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'Normal': prediction[0][0],
            'Pneumonia': prediction[0][1] 
        }
    }
    return jsonify(response)