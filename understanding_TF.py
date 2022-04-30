import os
import re
import zipfile

import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# x = tf.constant([1, 4])
# y = tf.constant([2, 5])
# z = tf.constant([3, 6])
# tf.stack([x, y, z])

# #add each one as new row
# tf.stack([x, y, z], axis=1)
#
# #add each one as new column
# tf.stack([x, y, z], axis=0)
#
# #add each one as new row (inverse order)
# tf.stack([x, y, z], axis=-1)
