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

testdata_path = '/home/ilya/gryaz/prj.scripts/mxnet/traffic_lights/testdata'
dataset_path = '/datasets/traffic_lights/sol_test'
train_path = os.path.join(dataset_path, 'train')

ctx = [mx.cpu()]
model_name = 'ResNet50_v2'
tuned_net = get_model(model_name, pretrained=True)
with tuned_net.name_scope():
    tuned_net.output = nn.Dense(2)
tuned_net.output.initialize(init.Xavier(), ctx = ctx)
tuned_net.collect_params().reset_ctx(ctx)
tuned_net.hybridize()
tuned_net.load_parameters('training_logs/ttl_v4__resnset20/params/two_traffic_lights_v4__resnet20_v2.params')

transform_test = transforms.Compose([
    transforms.Resize(100),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

for im_fname in os.listdir(testdata_path):
    im_fname = os.path.join(testdata_path, im_fname)

    img = image.imread(im_fname)

    plt.imshow(img.asnumpy())
    plt.show()

    img = transform_test(img)

    plt.imshow(nd.transpose(img, (1,2,0)).asnumpy())
    plt.show()

    pred = tuned_net(img.expand_dims(axis=0))

    class_names = os.listdir(train_path)
    print(class_names)
    ind = nd.argmax(pred, axis=1).astype('int')
    print('The input picture is classified as [%s], with probability %.3f.'%
          (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))
