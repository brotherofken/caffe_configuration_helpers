import caffe
from caffe import layers as layers
from caffe import params as params
from caffe.pycaffe import Net

# Class representing Caffe network
class Network():

    def __is_sequence(self, arg):
        return (not hasattr(arg, "strip") and 
                hasattr(arg, "__getitem__") or
                hasattr(arg, "__iter__"))


    def __init__(self, batch_size = 32, shape = (32, 32)):
        # Counter for layers of different types, e.g. conv, relu, pool.
        self.counters = dict()

        self.n = caffe.NetSpec()
        # Dummy data layer must be edited manually in prototxt
        self.n.data, self.n.label = layers.DummyData(shape=[dict(dim=[batch_size, 1, shape[0], shape[1]]),
                                                            dict(dim=[batch_size, 1, 1, 1])],
                                                     transform_param=dict(scale=1./255), ntop = 2)


    def blob_data(self):
        return self.n.data


    def blob_label(self):
        return self.n.label

    def add_layer(self, bottom, layer_type, counter, **kwargs):
        """
        Append new layer to network.
            bottom - previous layer
            layer_type - caffe layer type name (Convolution, Dropout, Pooling, ReLU, LRN, etc.)
            counter - name of counter to increment
        """
        suffix = "" if self.counters.get(counter) == None else str(self.counters.get(counter, 1))
        name = counter + suffix
        layer = None
        if self.__is_sequence(bottom):
            layer = layers.__getattr__(layer_type)(*bottom, **kwargs)
        else:
            layer = layers.__getattr__(layer_type)(bottom, **kwargs)
        self.n.__setattr__(name, layer)
        self.counters[counter] = self.counters.get(counter, 1) + 1
        return layer

    def add_dropout(self, bottom, counter = 'drop', in_place = True, **kwargs):
        return self.add_layer(bottom, 'Dropout', counter, in_place = in_place, **kwargs)


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
        suffix = "" if self.counters.get(counter) == None else str(self.counters.get(counter, 1))
        name = counter + suffix
        layer = layers.Concat(*bottom_layers, **kwargs)
        self.n.__setattr__(name, layer)
        self.counters[counter] = self.counters.get(counter, 1) + 1
        return layer


    def add_accuracy(self, bottom, counter = 'accuracy', **kwargs):
        return self.add_layer(bottom, 'Accuracy', counter, **kwargs)


    def add_inception_5(self, bottom, counter = 'inception5',
                        outs_1x1 = 64, 
                        outs_3x3_reduce = 96, outs_3x3 = 128,
                        outs_5x5_reduce = 16, outs_5x5 = 16,
                        outs_pool_proj = 32):
        """
        Append Inception-5 layer as described in Szegedy C. et al. Going deeper with convolutions
            You need to specify number of outputs for internal layers.
        """
        # Values that are common for conv layers
        common_params = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)]
        fill_xavier = dict(type='xavier')
        fill_const = dict(type='constant', value = 0.2)
        
        #
        conv_1x1 = self.add_convolution(bottom, counter = counter + '/1x1',
                                        num_output = outs_1x1, kernel_size = 1,
                                        param = common_params,
                                        weight_filler = fill_xavier,
                                        bias_filler = fill_const,
                                        )
        conv_1x1_relu = self.add_relu(conv_1x1, counter = counter + '/relu_1x1')
        
        # 3x3 branch : reduce -> relu -> conv -> relu
        reduce_3x3 = self.add_convolution(bottom, counter = counter + '/3x3_reduce',
                                          num_output = outs_3x3_reduce, kernel_size = 1,
                                          param = common_params,
                                          weight_filler = fill_xavier,
                                          bias_filler = fill_const,
                                          )
        reduce_3x3_relu = self.add_relu(reduce_3x3, counter = counter + '/relu_3x3_reduce')
        conv_3x3 = self.add_convolution(reduce_3x3_relu, counter = counter + '/3x3',
                                        num_output = outs_3x3, kernel_size = 3, pad = 1,
                                        param = common_params,
                                        weight_filler = fill_xavier,
                                        bias_filler = fill_const,
                                        )
        conv_3x3_relu = self.add_relu(conv_3x3, counter = counter + '/relu_3x3')
        
        # 5x5 branch : reduce -> relu -> conv -> relu
        reduce_5x5 = self.add_convolution(bottom, counter = counter + '/5x5_reduce',
                                          num_output = outs_5x5_reduce, kernel_size = 1,
                                          param = common_params,
                                          weight_filler = fill_xavier,
                                          bias_filler = fill_const,
                                          )
        reduce_5x5_relu = self.add_relu(reduce_5x5, counter = counter + '/relu_5x5_reduce')
        conv_5x5 = self.add_convolution(reduce_5x5_relu, counter = counter + '/5x5',
                                        num_output = outs_5x5, kernel_size = 5, pad = 2,
                                        param = common_params,
                                        weight_filler = fill_xavier,
                                        bias_filler = fill_const,
                                        )
        conv_5x5_relu = self.add_relu(conv_5x5, counter = counter + '/relu_5x5')
        
        # pool branch: pool-> projection -> relu
        pool_max_3x3 = self.add_pooling(bottom, counter = counter + '/pool',
                                        pool = params.Pooling.MAX,
                                        kernel_size = 3, stride = 1, pad = 1,
                                        )
        pool_max_3x3_proj = self.add_convolution(pool_max_3x3, counter = counter + '/pool_proj',
                                                 num_output = outs_pool_proj, kernel_size = 1,
                                                 param = common_params,
                                                 weight_filler = fill_xavier,
                                                 bias_filler = fill_const,
                                                 )
        pool_max_3x3_relu = self.add_relu(pool_max_3x3_proj, counter = counter + '/relu_pool_proj')
        
        concatentation = [conv_1x1_relu,conv_3x3_relu,conv_5x5_relu,pool_max_3x3_relu]
        concat = self.add_concat(concatentation, counter = counter + '/output')
        return concat

    def __str__(self):
        return str(self.n.to_proto())


