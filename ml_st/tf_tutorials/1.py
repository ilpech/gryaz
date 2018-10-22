import matplotlib.pyplot as plt
import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

from mnist import MNIST
data = input_data.read_data_sets('data/MNIST', one_hot = True)

eval_labels = np.asarray(data.test.labels, dtype=np.int32)

print(data.test.labels)


tf.logging.set_verbosity(old_v)
