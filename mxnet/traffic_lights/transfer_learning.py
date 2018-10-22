import mxnet as mx
import numpy as np
import os, time, shutil

import matplotlib.pyplot as plt
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.model_zoo import get_model

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

num_gpus = 1
num_workers = 8
ctx = [mx.gpu()]
per_device_batch_size = 4
batch_size = per_device_batch_size * max(num_gpus, 1)

print("Batch size", batch_size)

jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(100),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.Resize(100),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

classes = 2

epochs = 240
lr = 0.1
momentum = 0.9
wd = 0.0001
lr_decay = 0.1
lr_decay_epoch = [80, 160, np.inf]
lr_decay_count = 0


dataset_path = '/datasets/traffic_lights/sol_test'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

model_name = 'ResNet50_v2'
# model_name = 'alexnet'
tuned_net = get_model(model_name, pretrained=True)
with tuned_net.name_scope():
    tuned_net.output = nn.Dense(classes)
tuned_net.output.initialize(init.Xavier(), ctx = ctx)
tuned_net.collect_params().reset_ctx(ctx)
tuned_net.hybridize()

trainer = gluon.Trainer(tuned_net.collect_params(), 'nag', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
train_metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()

train_history = TrainingHistory(['training-error', 'validation-error'])


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

    if epoch % 10 == 0:
        tuned_net.save_parameters('ttl_epoch_{:03d}__resnet20_v.params'.format(epoch))


train_history.plot()

name, test_acc = test(tuned_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))
