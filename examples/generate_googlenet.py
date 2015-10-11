import caffe
from caffe import layers as layers
from caffe import params as params
from caffe.pycaffe import Net

import network

# In[]:

googlenet = network.Network(32)
l_conv1 = googlenet.add_convolution(googlenet.blob_data())
l_drop1 = googlenet.add_dropout(l_conv1, dropout_ratio = 0.5)
l_conv2 = googlenet.add_convolution(l_drop1, num_output = 32, weight_filler=dict(type='Xavier'))
l_pool1 = googlenet.add_pooling(l_conv2, pool=params.Pooling.MAX)
l_relu1 = googlenet.add_relu(l_pool1)
l_inception5_1 = googlenet.add_inception_5(l_relu1)

print(str(googlenet))

# In[]:

#def googlenet(batch_size):

batch_size = 32

n = caffe.NetSpec()
n.data, n.label = layers.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                          dict(dim=[batch_size, 1, 1, 1])],
                              transform_param=dict(scale=1./255), ntop=2)

n.conv1 = layers.Convolution(n.data, kernel_size=5, num_output=20,
                             weight_filler=dict(type='xavier'))
n.pool1 = layers.Pooling(n.conv1, kernel_size=2, stride=2, pool=params.Pooling.MAX)

n.conv2 = layers.Convolution(n.pool1, kernel_size=5, num_output=50,
                             weight_filler=dict(type='xavier'))
n.pool2 = layers.Pooling(n.conv2, kernel_size=2, stride=2, pool=params.Pooling.MAX)

n.ip1 = layers.InnerProduct(n.pool2, num_output=500,
                            weight_filler=dict(type='xavier'))
n.relu1 = layers.ReLU(n.ip1, in_place=True)

n.ip2 = layers.InnerProduct(n.relu1, num_output=10,
                            weight_filler=dict(type='xavier'))

n.concat = layers.Concat(n.pool2,n.ip2)

n.loss = layers.SoftmaxWithLoss(n.concat)

# In[]:
print n.to_proto()