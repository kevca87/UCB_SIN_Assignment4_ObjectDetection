import os
import re
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify, send_file
from markupsafe import escape
from RetinaNet import *

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

num_classes = 80
batch_size = 2

## Setting up training parameters

model_dir = "retinanet/"
label_encoder = LabelEncoder()

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
boundaries=learning_rate_boundaries, values=learning_rates )

## Initializing and compiling model

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

## Setting up callbacks

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

## Loading weights

# Change this to `model_dir` when not using the downloaded weights
weights_dir = "data"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

## Building inference model

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

int2str = dataset_info.features["objects"]["label"].int2str

app = Flask(__name__)

@app.route('/objectdetection', methods=['POST'])
def object_detection():

    print('\n\n------------------------------')
    print('Image arriving')

    image = request.files.get('image', '')
    image = Image.open(image)
    image = np.array(image, dtype=np.uint8)
    print(type(image))
    print(image.shape)

    input_image, ratio = prepare_image(image)
    print('\n\n------------------------------')
    print('Predict')
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [ int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections] ]
    detected_image = visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

    print('\n\n------------------------------')
    print('Detected')
    print(type(detected_image))
    print(detected_image.shape)

    file_name = 'art.png'
    Image.fromarray(detected_image).convert("RGB").save(file_name)
    #send_file(image)

    return 'OK', 201

#Python asigna el valor __main__ a la variable __name__ cuando se ejecuta en modo standalone
if __name__ == '__main__':
    app.run(port=8080, debug=False)