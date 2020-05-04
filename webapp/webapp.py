#!/usr/bin/env python3

##
## This requires 'bottle' to be installed (from pip)
##

from bottle import route, request, run, template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# dimensions of our images
img_width, img_height = 200, 200

# load the model we saved
MODEL_PATH = 'model.h5'

model = load_model(MODEL_PATH)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
from bottle import route, run, template, static_file, post

import tensorflow as tf
from tensorflow import keras

MODEL_PATH = 'model.h5'

@route('/')
def index():
    return template('index', {})

@route('/static/<path>')
def static(path):
    return static_file(path, root='./static')

@route('/predict', method='POST')
def do_predict():
    data = request.files.data
    name, ext = os.path.splitext(data.filename)
    print(name, ext)
    if ext not in ('.png', '.jpg', '.jpeg'):
        return 'File extension not allowed.'

    if data.file:
        img = image.load_img(data.file, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32') / 255
        res = model.predict(x)
        return 'Probability not ad: {}, Probability ad: {}'.format(res[0][0], res[0][1])

run(host='localhost', port=8000)
