import os
import re
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#from RetinaNet import *
from PIL import Image


#http://flask.palletsprojects.com/en/1.1.x/
#http://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart
from flask import Flask, request, jsonify, send_file
from markupsafe import escape

app = Flask(__name__)

# @app.route('/user/<username>')
# def show_user_profile(username):
#     # show the user profile for that user
#     return 'User %s' % escape(username)

# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     # show the post with the given id, the id is an integer
#     return 'Post %d' % post_id

# @app.route('/path/<path:subpath>')
# def show_subpath(subpath):
#     # show the subpath after /path/
#     return 'Subpath %s' % escape(subpath)

# @app.route('/api', methods=['POST'])
# def echo():
#     data = request.get_json(force=True)
#     print(data)
#     print(type(data))
#     return jsonify(data), 201

# @app.route('/api/lol', methods=['PUT'])
# def echo_lol():
#     data = request.get_json(force=True)
#     print(data)
#     print(type(data))
#     return jsonify(data), 201

# @app.route('/diabetes/api/predict', methods=['GET'])
# def diabetes_predict():
#     data = request.get_json(force=True)
#     print(data)
#     print(type(data))
#     return jsonify(data), 201

@app.route('/objectdetection', methods=['POST'])
def object_detection():
    image = request.files.get('image', '')
    image = Image.open(image)
    image = np.array(image, dtype=np.uint8)
    print(type(image))
    print(image.shape)
    #send_file(image)
    return 'OK', 201

#Python asigna el valor __main__ a la variable __name__ cuando se ejecuta en modo standalone
if __name__ == '__main__':
    app.run(port=8080, debug=True)