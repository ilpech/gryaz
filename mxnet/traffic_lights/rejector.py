import mxnet as mx
import numpy as np
import os, time

import matplotlib.pyplot as plt
from mxnet import gluon, image, init
from mxnet.gluon import nn
from gluoncv.utils import TrainingHistory
from gluoncv.model_zoo import get_model

from tools import *

dataset_path = '/datasets/traffic_lights/dataset_rejector_splitted'

demo = False
num_gpus = 1
num_workers = 8
ctx = [mx.gpu()]
per_device_batch_size = 1
batch_size = per_device_batch_size * max(num_gpus, 1)

classes = 3
model_name = 'ResNet50_v2'
tuned_net = get_model(model_name, pretrained=True)
with tuned_net.name_scope():
    tuned_net.output = nn.Dense(classes)
tuned_net.output.initialize(init.Xavier(), ctx = ctx)
tuned_net.collect_params().reset_ctx(ctx)
tuned_net.hybridize()

if demo:
    epochs = 40
else:
    epochs = 240
lr = 0.1
momentum = 0.9
wd = 0.0001
lr_decay = 0.1
lr_decay_epoch = [80, 160, np.inf]
lr_decay_count = 0
optimizer = 'nag'
optimizer_params = {'wd': wd, 'momentum': momentum, 'learning_rate': lr}
trainer = gluon.Trainer(tuned_net.collect_params(), optimizer, optimizer_params)
train_metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()

train_history = TrainingHistory(['training-error', 'validation-error'])

train_data, val_data, test_data = get_data_raw(dataset_path, batch_size, num_workers)

print("Batch size", batch_size)
print('Workon dataset_path: ', dataset_path)
print('Model Name: ', model_name)
for epoch in range(epochs):
    tic = time.time()
    train_loss = 0
    train_metric.reset()

    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    for i, batch in enumerate(train_data):
        # Extract data and label
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        # AutoGrad
        with ag.record():
            outputs = [tuned_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size)

        train_loss += sum([l.sum().asscalar() for l in loss])
        train_metric.update(label, outputs)

    name, train_acc = train_metric.get()
    name, val_acc = test(tuned_net, val_data, ctx)

    train_history.update([1-train_acc, 1-val_acc])

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, time.time() - tic))

    if (epoch+1) % 10 == 0:
        print('Params saved on epoch #', epoch)
        tuned_net.save_parameters('rejector2_{:03d}__resnet50_v.params'.format(epoch))
train_history.plot()

name, test_acc = test(tuned_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))
