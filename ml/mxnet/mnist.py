#done on https://mxnet.incubator.apache.org/tutorials/python/mnist.html

import mxnet as mx

mnist = mx.test_utils.get_mnist()

mx.random.seed(42)

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter   = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

data = mx.sym.var('data')
#from 4D shape to 2D (batch_size, channels*hgt*wdth)
data = mx.sym.flatten(data=data)

#fc - fully connected layers
#act - activate functions

fc1 = mx.sym.FullyConnected(data=data,num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type='relu')

fc2 = mx.sym.FullyConnected(data=act1, num_hidden=128)
act2 = mx.sym.Activation(data=fc2, atc_type='relu')

fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
