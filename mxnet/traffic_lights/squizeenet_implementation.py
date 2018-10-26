# Demo mode uses the validation dataset for training, which is smaller and faster to train.
demo = False
log_interval = 100

# Options are imperative or hybrid. Use hybrid for better performance.
mode = 'hybrid'

# training hyperparameters
batch_size = 10
if demo:
    epochs = 5
    learning_rate = 0.02
    wd = 0.002
else:
    epochs = 40
    learning_rate = 0.05
    wd = 0.002

# the class weight for hotdog class to help the imbalance problem.
positive_class_weight = 1

import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from collections import OrderedDict
import skimage.io as io

import mxnet as mx
from mxnet.test_utils import download
mx.random.seed(127)

# setup the contexts; will use gpus if avaliable, otherwise cpu
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]

dataset_path = '/datasets/traffic_lights/dataset_rec'
train = os.path.join(dataset_path, 'dataset_rec.rec')
val   = os.path.join(dataset_path, 'dataset_rec_val.rec')

dataset_files = {
    'train': (train),
    'validation': (val)
    }

if demo:
    training_dataset = dataset_files['validation']
else:
    training_dataset = dataset_files['train']

validation_dataset = dataset_files['validation']

train_iter = mx.io.ImageRecordIter(path_imgrec=training_dataset,
                                   min_img_size=256,
                                   data_shape=(3, 224, 224),
                                   rand_crop=True,
                                   shuffle=True,
                                   batch_size=batch_size,
                                   max_random_scale=1.5,
                                   min_random_scale=0.75,
                                   rand_mirror=True)
val_iter = mx.io.ImageRecordIter(path_imgrec=validation_dataset,
                                 min_img_size=256,
                                 data_shape=(3, 224, 224),
                                 batch_size=batch_size)


from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

net = models.squeezenet1_1(pretrained=True, prefix='rejector_', ctx=ctx)

rejector_net = models.squeezenet1_1(prefix='rejector_', classes=3)
rejector_net.collect_params().initialize(ctx=ctx)
rejector_net.features = net.features

# return metrics string representation
def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])
metric = mx.metric.create(['acc'])

import mxnet.gluon as gluon
from mxnet.image import color_normalize

def evaluate(net, data_iter, ctx):
    data_iter.reset()
    for batch in data_iter:
        data = color_normalize(batch.data[0]/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    out = metric.get()
    metric.reset()
    return out

import mxnet.autograd as autograd

def train(net, train_iter, val_iter, epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': wd})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # best_f1 = 0
    val_names, val_accs = evaluate(net, val_iter, ctx)
    logging.info('[Initial] validation: %s'%(metric_str(val_names, val_accs)))
    for epoch in range(epochs):
        tic = time.time()
        train_iter.reset()
        btic = time.time()
        for i, batch in enumerate(train_iter):
            # the model zoo models expect normalized images
            print(batch.data[0])
            data = color_normalize(batch.data[0]/255,
                                   mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                                   std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    # rescale the loss based on class to counter the imbalance problem
                    L = loss(z, y) * (1+y*positive_class_weight)/positive_class_weight
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if log_interval and not (i+1)%log_interval:
                names, accs = metric.get()
                logging.info('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                               epoch, i, batch_size/(time.time()-btic), metric_str(names, accs)))
            btic = time.time()

        names, accs = metric.get()
        metric.reset()
        logging.info('[Epoch %d] training: %s'%(epoch, metric_str(names, accs)))
        # logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        # val_names, val_accs = evaluate(net, val_iter, ctx)
        # logging.info('[Epoch %d] validation: %s'%(epoch, metric_str(val_names, val_accs)))

        # if val_accs[1] > best_f1:
        #     best_f1 = val_accs[1]
        #     logging.info('Best validation f1 found. Checkpointing...')
        #     net.save_parameters('deep-dog-%d.params'%(epoch))

if mode == 'hybrid':
    rejector_net.hybridize()
if epochs > 0:
    rejector_net.collect_params().reset_ctx(ctx)
    train(rejector_net, train_iter, val_iter, epochs, ctx)

print('g')
