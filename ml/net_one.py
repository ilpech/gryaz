#https://habr.com/post/271563/
import numpy as np


training_set_X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
training_set_Y  = np.array([[0,1,1,0]]).T

np.random.seed(1)

weights_X  = 2*np.random.random((3,4)) - 1
weights_Y  = 2*np.random.random((4,1)) - 1

for set in range(60000):
     layer_X = 1/(1 + np.exp(-(np.dot(training_set_X, weights_X))))
     layer_Y = 1/(1 + np.exp(-(np.dot(layer_X, weights_Y))))

     layer_Y_delta = (training_set_Y - layer_Y)*(layer_Y * (1 - layer_Y))
     layer_X_delta = layer_Y_delta.dot(weights_Y.T) * (layer_X * (1-layer_X))

     weights_Y += layer_X.T.dot(layer_Y_delta)
     weights_X += training_set_X.T.dot(layer_X_delta)

print("result ", layer_Y)
print("expected ", training_set_Y)
