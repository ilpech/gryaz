#done on https://mxnet.incubator.apache.org/tutorials/python/mnist.html

import mxnet as mx

mnist = mx.test_utils.get_mnist()

mx.random.seed(42)

# ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
ctx = mx.cpu()

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
act2 = mx.sym.Activation(data=fc2, act_type='relu')

fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')


import logging
logging.getLogger().setLevel(logging.DEBUG)

mlp_model = mx.mod.Module(symbol=mlp,context=ctx)
mlp_model.fit(
    train_iter,  # train data
    eval_data=val_iter,  # validation data
    optimizer='sgd',  # use SGD to train
    optimizer_params={'learning_rate':0.1},  # use fixed learning rate
    eval_metric='acc',  # report accuracy during training
    batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
    num_epoch=10
)

# After the above training completes, we can evaluate the trained model by running predictions on test data.
# The following source code computes the prediction probability scores for each test image.
# prob[i][j] is the probability that the i-th test image contains the j-th output class

test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]
