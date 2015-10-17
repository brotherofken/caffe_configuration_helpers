import sys
sys.path.insert(0, '..')

from caffe import params as params
import caffe
import network

# In[]: Parameters common for all convolutional layers
common_params = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)]
fill_xavier = dict(type='xavier')
fill_const = dict(type='constant', value = 0.2)
conv_kwargs = dict(param = common_params, weight_filler = fill_xavier, bias_filler = fill_const)

inception3a_kwargs = dict(outs_1x1 = 64, outs_3x3_reduce = 64, outs_3x3 = 64, outs_double_3x3_reduce = 64, outs_double_3x3 = 96, outs_pool_proj = 32, outs_pool_proj_type = params.Pooling.AVE)
inception3b_kwargs = dict(outs_1x1 = 64, outs_3x3_reduce = 64, outs_3x3 = 96, outs_double_3x3_reduce = 64, outs_double_3x3 = 96, outs_pool_proj = 64, outs_pool_proj_type = params.Pooling.AVE)
inception3c_kwargs = dict(outs_3x3_reduce = 96, outs_3x3 = 160, outs_double_3x3_reduce = 64, outs_double_3x3 = 96, reduction_stride = 2, reduction_inception = True)

inception4a_kwargs = dict(outs_1x1 = 224, outs_3x3_reduce = 64, outs_3x3 = 96, outs_double_3x3_reduce = 96, outs_double_3x3 = 128, outs_pool_proj = 128, outs_pool_proj_type = params.Pooling.AVE)
inception4b_kwargs = dict(outs_1x1 = 192, outs_3x3_reduce = 96, outs_3x3 = 128, outs_double_3x3_reduce = 96, outs_double_3x3 = 128, outs_pool_proj = 128, outs_pool_proj_type = params.Pooling.AVE)
inception4c_kwargs = dict(outs_1x1 = 160, outs_3x3_reduce = 128, outs_3x3 = 160, outs_double_3x3_reduce = 128, outs_double_3x3 = 160, outs_pool_proj = 128, outs_pool_proj_type = params.Pooling.AVE)
inception4d_kwargs = dict(outs_1x1 = 96, outs_3x3_reduce = 128, outs_3x3 = 192, outs_double_3x3_reduce = 160, outs_double_3x3 = 192, outs_pool_proj = 128, outs_pool_proj_type = params.Pooling.AVE)
inception4e_kwargs = dict(outs_3x3_reduce = 96, outs_3x3 = 192, outs_double_3x3_reduce = 192, outs_double_3x3 = 256, reduction_stride = 2, pool_pad = 0, reduction_inception = True)

inception5a_kwargs = dict(outs_1x1 = 352, outs_3x3_reduce = 192, outs_3x3 = 320, outs_double_3x3_reduce = 160, outs_double_3x3 = 224, outs_pool_proj = 128, outs_pool_proj_type = params.Pooling.AVE)
inception5b_kwargs = dict(outs_1x1 = 352, outs_3x3_reduce = 192, outs_3x3 = 320, outs_double_3x3_reduce = 192, outs_double_3x3 = 224, outs_pool_proj = 128, outs_pool_proj_type = params.Pooling.MAX)


# Network configuration 
googlenet = network.Network(32, shape = (224, 224))

data = googlenet.blob_data()
label = googlenet.blob_label()

l_conv1 = googlenet.add_convolution(data, counter = 'conv1', num_output = 64, pad = 3, kernel_size = 7, stride = 2, **conv_kwargs)
l_conv1_relu1 = googlenet.add_relu(l_conv1, counter = 'conv1_relu')
l_pool1 = googlenet.add_pooling(l_conv1_relu1, counter = 'pool1', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)

l_conv2 = googlenet.add_convolution(l_pool1, counter = 'conv2', num_output = 192, kernel_size = 3, stride = 1, **conv_kwargs)
l_conv2_relu = googlenet.add_relu(l_conv2, counter = 'conv2_relu')
l_pool2 = googlenet.add_pooling(l_conv2_relu, counter = 'pool2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)

l_inception6_3a = googlenet.add_inception_6(l_pool2, counter = 'inception_3a', **inception3a_kwargs)
l_inception6_3b = googlenet.add_inception_6(l_inception6_3a, counter = 'inception_3b', **inception3b_kwargs)
l_inception6_3c = googlenet.add_inception_6(l_inception6_3b, counter = 'inception_3c', **inception3c_kwargs)

l_inception6_4a = googlenet.add_inception_6(l_inception6_3c, counter = 'inception_4a', **inception4a_kwargs)
l_inception6_4b = googlenet.add_inception_6(l_inception6_4a, counter = 'inception_4b', **inception4b_kwargs)
l_inception6_4c = googlenet.add_inception_6(l_inception6_4b, counter = 'inception_4c', **inception4c_kwargs)
l_inception6_4d = googlenet.add_inception_6(l_inception6_4c, counter = 'inception_4d', **inception4d_kwargs)
l_inception6_4e = googlenet.add_inception_6(l_inception6_4d, counter = 'inception_4e', **inception4e_kwargs)

l_inception6_5a = googlenet.add_inception_6(l_inception6_4e, counter = 'inception_5a', **inception5a_kwargs)
l_inception6_5b = googlenet.add_inception_6(l_inception6_5a, counter = 'inception_5b', **inception5b_kwargs)

l_pool5 = googlenet.add_pooling(l_inception6_5b, counter = 'pool_7x7', pool = params.Pooling.AVE, kernel_size = 7, stride = 1)

l_loss_drop = googlenet.add_dropout(l_pool5, counter = 'loss/drop', dropout_ratio = 0.4)

l_loss_clf = googlenet.add_dense(l_loss_drop, counter = 'loss/classifier',
                              num_output = 1000,
                              param = common_params, weight_filler = fill_xavier,
                              bias_filler = dict(type='constant', value = 0))

l_softmax = googlenet.add_softmax_with_loss([l_loss_clf, label], counter = 'loss/loss')
l_loss1_accuracy1 = googlenet.add_accuracy([l_loss_clf, label], counter = 'loss/top-1', include = dict(phase = 1))
l_loss1_accuracy5 = googlenet.add_accuracy([l_loss_clf, label], counter = 'loss/top-5', include = dict(phase = 1), accuracy_param=dict(top_k=5))

print(str(googlenet))

text_file = open("ilsvrc2015_inception6.prototxt", "w")
text_file.write('name: \"Inception6\"\n')
text_file.write(str(googlenet))
text_file.close()

# In[]: Print network information
caffe.set_mode_cpu()
net = caffe.Net('ilsvrc2015_inception6.prototxt', caffe.TEST)
#net = caffe.Net('googlenet.prototxt', caffe.TEST)

for k, v in net.blobs.items():
    print (k, v.data.shape)
