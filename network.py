import caffe
from caffe import layers as layers
from caffe import params as params
from caffe.pycaffe import Net

# Class representing Caffe network
class Network():
    net = None
    counters = dict()


    def __init__(self, batch_size):
        batch_size = 32
        self.n = caffe.NetSpec()
        self.n.data, self.n.label = layers.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                                            dict(dim=[batch_size, 1, 1, 1])],
                                                     transform_param=dict(scale=1./255), ntop=2)


    def blob_data(self):
        return self.n.data


    def blob_label(self):
        return self.n.label


    def add_layer(self, bottom, layer_type, counter, **kwargs):
        name = counter + str(self.counters.get(counter, 1))
        layer = layers.__getattr__(layer_type)(bottom, **kwargs)
        self.n.__setattr__(name, layer)
        self.counters[counter] = self.counters.get(counter, 1) + 1
        return layer

    def add_dropout(self, bottom, counter = 'drop', **kwargs):
        return self.add_layer(bottom, 'Dropout', counter, **kwargs)


    def add_convolution(self, bottom, counter = 'conv', **kwargs):
        return self.add_layer(bottom, 'Convolution', counter, **kwargs)


    def add_dense(self, bottom, counter = 'dense', **kwargs):
        return self.add_layer(bottom, 'InnerProduct', counter, **kwargs)


    def add_softmax_with_loss(self, bottom, counter = 'classifier', **kwargs):
        return self.add_layer(bottom, 'SoftmaxWithLoss', counter, **kwargs)


    def add_relu(self, bottom, counter = 'relu', in_place = True, **kwargs):
        return self.add_layer(bottom, 'ReLU', counter, in_place = in_place, **kwargs)

    def add_lrn(self, bottom, counter = 'lrn', **kwargs):
        return self.add_layer(bottom, 'LRN', counter, **kwargs)


    def add_pooling(self, bottom, counter = 'pool', **kwargs):
        return self.add_layer(bottom, 'Pooling', counter, **kwargs)

    def add_concat(self, bottom_layers, counter = 'concat', **kwargs):
        name = counter + str(self.counters.get(counter, 1))
        layer = layers.Concat(*bottom_layers, **kwargs)
        self.n.__setattr__(name, layer)
        self.counters[counter] = self.counters.get(counter, 1) + 1
        return layer

    def add_inception_5(self, bottom, counter = 'inception5',
                        outs_1x1 = 64, 
                        outs_3x3_reduce = 96, outs_3x3 = 128,
                        outs_5x5_reduce = 16, outs_5x5 = 16,
                        outs_pool_proj = 32):
        # Values that are common for conv layers
        common_params = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)]
        fill_xavier = dict(type='xavier')
        fill_const = dict(type='constant', value = 0.2)
        
        #
        conv_1x1 = self.add_convolution(bottom, counter = counter + '/1x1_',
                                        num_output = outs_1x1, kernel_size = 1,
                                        param = common_params,
                                        weight_filler = fill_xavier,
                                        bias_filler = fill_const,
                                        )
        conv_1x1_relu = self.add_relu(conv_1x1, counter = counter + '/relu_1x1_')
        
        # 3x3 branch : reduce -> relu -> conv -> relu
        reduce_3x3 = self.add_convolution(bottom, counter = counter + '/3x3_reduce_',
                                          num_output = outs_3x3_reduce, kernel_size = 1,
                                          param = common_params,
                                          weight_filler = fill_xavier,
                                          bias_filler = fill_const,
                                          )
        reduce_3x3_relu = self.add_relu(reduce_3x3, counter = counter + '/relu_3x3_reduce_')
        conv_3x3 = self.add_convolution(reduce_3x3_relu, counter = counter + '/conv_3x3_',
                                        num_output = outs_3x3, kernel_size = 3, pad = 1,
                                        param = common_params,
                                        weight_filler = fill_xavier,
                                        bias_filler = fill_const,
                                        )
        conv_3x3_relu = self.add_relu(conv_3x3, counter = counter + '/conv_3x3_relu_')
        
        # 5x5 branch : reduce -> relu -> conv -> relu
        reduce_5x5 = self.add_convolution(bottom, counter = counter + '/5x5_reduce_',
                                          num_output = outs_5x5_reduce, kernel_size = 1,
                                          param = common_params,
                                          weight_filler = fill_xavier,
                                          bias_filler = fill_const,
                                          )
        reduce_5x5_relu = self.add_relu(reduce_5x5, counter = counter + '/relu_5x5_reduce_')
        conv_5x5 = self.add_convolution(reduce_5x5_relu, counter = counter + '/5x5_',
                                        num_output = outs_5x5, kernel_size = 5, pad = 2,
                                        param = common_params,
                                        weight_filler = fill_xavier,
                                        bias_filler = fill_const,
                                        )
        conv_5x5_relu = self.add_relu(conv_5x5, counter = counter + '/relu_5x5_')
        
        # pool branch: pool-> projection -> relu
        pool_max_3x3 = self.add_pooling(bottom, counter = counter + '/pool_',
                                        pool = params.Pooling.MAX,
                                        kernel_size = 3, stride = 1, pad = 1,
                                        )
        pool_max_3x3_proj = self.add_convolution(pool_max_3x3, counter = counter + '/pool_proj_',
                                                 num_output = outs_pool_proj, kernel_size = 1,
                                                 param = common_params,
                                                 weight_filler = fill_xavier,
                                                 bias_filler = fill_const,
                                                 )
        pool_max_3x3_relu = self.add_relu(pool_max_3x3_proj, counter = counter + '/relu_pool_proj_')
        
        concatentation = [conv_1x1_relu,conv_3x3_relu,conv_5x5_relu,pool_max_3x3_relu]
        concat = self.add_concat(concatentation, counter = counter + '/output')
        return concat

    def __str__(self):
        return str(self.n.to_proto())

# In[]: Showcase

import network

common_params = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)]
fill_xavier = dict(type='xavier')
fill_const = dict(type='constant', value = 0.2)

conv_kwargs = dict(param = common_params, weight_filler = fill_xavier, bias_filler = fill_const)

inception3a_kwargs = dict(outs_1x1 = 64, outs_3x3_reduce = 96,  outs_3x3 = 128, outs_5x5_reduce = 16, outs_5x5 = 32, outs_pool_proj = 32)
inception3b_kwargs = dict(outs_1x1 = 128, outs_3x3_reduce = 128, outs_3x3 = 128, outs_5x5_reduce = 32, outs_5x5 = 96, outs_pool_proj = 64)

inception4a_kwargs = dict(outs_1x1 = 192, outs_3x3_reduce = 96,  outs_3x3 = 208, outs_5x5_reduce = 16, outs_5x5 = 48, outs_pool_proj = 64)
inception4b_kwargs = dict(outs_1x1 = 160, outs_3x3_reduce = 112, outs_3x3 = 224, outs_5x5_reduce = 24, outs_5x5 = 64, outs_pool_proj = 64)
inception4c_kwargs = dict(outs_1x1 = 128, outs_3x3_reduce = 128, outs_3x3 = 256, outs_5x5_reduce = 24, outs_5x5 = 64, outs_pool_proj = 64)
inception4d_kwargs = dict(outs_1x1 = 112, outs_3x3_reduce = 144, outs_3x3 = 288, outs_5x5_reduce = 32, outs_5x5 = 64, outs_pool_proj = 64)
inception4e_kwargs = dict(outs_1x1 = 256, outs_3x3_reduce = 160, outs_3x3 = 320, outs_5x5_reduce = 32, outs_5x5 = 128, outs_pool_proj = 128)

inception5a_kwargs = dict(outs_1x1 = 256, outs_3x3_reduce = 160,  outs_3x3 = 320, outs_5x5_reduce = 32, outs_5x5 = 128, outs_pool_proj = 128)
inception5b_kwargs = dict(outs_1x1 = 384, outs_3x3_reduce = 192, outs_3x3 = 384, outs_5x5_reduce = 48, outs_5x5 = 128, outs_pool_proj = 128)


googlenet = Network(32)

l_conv1 = googlenet.add_convolution(googlenet.blob_data(), counter = 'conv1/7x7_s2',
                                    num_output = 64, pad = 3, kernel_size = 7, stride = 2,
                                    **conv_kwargs)
l_relu1 = googlenet.add_relu(l_conv1)
l_pool1 = googlenet.add_pooling(l_relu1, counter = 'pool1/3x3_s2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)
l_lrn1 = googlenet.add_lrn(l_pool1, counter = 'pool1/norm', local_size = 5, alpha = 0.0001, beta = 0.75)

l_conv2_reduce = googlenet.add_convolution(l_lrn1, counter = 'conv2/3x3_reduce', num_output = 64, kernel_size = 1, **conv_kwargs)
l_conv2_reduce_relu = googlenet.add_relu(l_conv2_reduce)
l_conv2_3x3 =googlenet.add_convolution(l_conv2_reduce_relu, counter = 'conv2/3x3', num_output = 192, pad = 3, kernel_size = 3, **conv_kwargs)
l_conv2_3x3_relu = googlenet.add_relu(l_conv2_3x3)

l_lrn2 = googlenet.add_lrn(l_conv2_3x3_relu, counter = 'conv2/norm2', local_size = 5, alpha = 0.0001, beta = 0.75)
l_pool2 = googlenet.add_pooling(l_lrn2, counter = 'pool2/3x3_s2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)

l_inception5_3a = googlenet.add_inception_5(l_pool2, counter = 'inception_3a', **inception3a_kwargs)
l_inception5_3b = googlenet.add_inception_5(l_inception5_3a, counter = 'inception_3b', **inception3b_kwargs)

l_pool3 = googlenet.add_pooling(l_inception5_3b, counter = 'pool3/3x3_s2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)

l_inception5_4a = googlenet.add_inception_5(l_pool3, counter = 'inception_4a', **inception4a_kwargs)
l_inception5_4b = googlenet.add_inception_5(l_inception5_4a, counter = 'inception_4b', **inception4b_kwargs)
l_inception5_4c = googlenet.add_inception_5(l_inception5_4b, counter = 'inception_4c', **inception4c_kwargs)
l_inception5_4d = googlenet.add_inception_5(l_inception5_4c, counter = 'inception_4d', **inception4d_kwargs)
l_inception5_4e = googlenet.add_inception_5(l_inception5_4d, counter = 'inception_4e', **inception4e_kwargs)

l_pool4 = googlenet.add_pooling(l_inception5_4e, counter = 'pool4/3x3_s2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)

l_inception5_5a = googlenet.add_inception_5(l_pool4, counter = 'inception_5a', **inception5a_kwargs)
l_inception5_5b = googlenet.add_inception_5(l_inception5_5a, counter = 'inception_5b', **inception5b_kwargs)

l_pool5 = googlenet.add_pooling(l_inception5_5b, counter = 'pool5/7x7_s1', pool = params.Pooling.AVE, kernel_size = 7, stride = 1)

l_drop1 = googlenet.add_dropout(l_pool5, counter = 'pool5/drop_7x7_s1', dropout_ratio = 0.4)

l_dense1 = googlenet.add_dense(l_drop1, counter = 'loss3/classifier',
                              num_output = 1000,
                              param = common_params, weight_filler = fill_xavier,
                              bias_filler = dict(type='constant', value = 0))

l_softmax = googlenet.add_softmax_with_loss(l_dense1, loss_weight = 1)

print(str(googlenet))

text_file = open("googlenet_text.txt", "w")
text_file.write(str(googlenet))
text_file.close()