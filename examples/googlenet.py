# In[]: Showcase

import sys
sys.path.insert(0, '..')

from caffe import params as params
import network

common_params = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)]
fill_xavier = dict(type='xavier')
fill_const = dict(type='constant', value = 0.2)

conv_kwargs = dict(param = common_params, weight_filler = fill_xavier, bias_filler = fill_const)

inception3a_kwargs = dict(outs_1x1 = 64, outs_3x3_reduce = 96,  outs_3x3 = 128, outs_5x5_reduce = 16, outs_5x5 = 32, outs_pool_proj = 32)
inception3b_kwargs = dict(outs_1x1 = 128, outs_3x3_reduce = 128, outs_3x3 = 192, outs_5x5_reduce = 32, outs_5x5 = 96, outs_pool_proj = 64)

inception4a_kwargs = dict(outs_1x1 = 192, outs_3x3_reduce = 96,  outs_3x3 = 208, outs_5x5_reduce = 16, outs_5x5 = 48, outs_pool_proj = 64)
inception4b_kwargs = dict(outs_1x1 = 160, outs_3x3_reduce = 112, outs_3x3 = 224, outs_5x5_reduce = 24, outs_5x5 = 64, outs_pool_proj = 64)
inception4c_kwargs = dict(outs_1x1 = 128, outs_3x3_reduce = 128, outs_3x3 = 256, outs_5x5_reduce = 24, outs_5x5 = 64, outs_pool_proj = 64)
inception4d_kwargs = dict(outs_1x1 = 112, outs_3x3_reduce = 144, outs_3x3 = 288, outs_5x5_reduce = 32, outs_5x5 = 64, outs_pool_proj = 64)
inception4e_kwargs = dict(outs_1x1 = 256, outs_3x3_reduce = 160, outs_3x3 = 320, outs_5x5_reduce = 32, outs_5x5 = 128, outs_pool_proj = 128)

inception5a_kwargs = dict(outs_1x1 = 256, outs_3x3_reduce = 160,  outs_3x3 = 320, outs_5x5_reduce = 32, outs_5x5 = 128, outs_pool_proj = 128)
inception5b_kwargs = dict(outs_1x1 = 384, outs_3x3_reduce = 192, outs_3x3 = 384, outs_5x5_reduce = 48, outs_5x5 = 128, outs_pool_proj = 128)

googlenet = None
googlenet = network.Network(32)

l_conv1 = googlenet.add_convolution(googlenet.blob_data(), counter = 'conv1/7x7_s2',
                                    num_output = 64, pad = 3, kernel_size = 7, stride = 2,
                                    **conv_kwargs)
l_relu1 = googlenet.add_relu(l_conv1, counter = 'conv1/relu_7x7')
l_pool1 = googlenet.add_pooling(l_relu1, counter = 'pool1/3x3_s2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)
l_lrn1 = googlenet.add_lrn(l_pool1, counter = 'pool1/norm', local_size = 5, alpha = 0.0001, beta = 0.75)

l_conv2_reduce = googlenet.add_convolution(l_lrn1, counter = 'conv2/3x3_reduce', num_output = 64, kernel_size = 1, **conv_kwargs)
l_conv2_reduce_relu = googlenet.add_relu(l_conv2_reduce, counter = 'conv2/relu_3x3_reduce')
l_conv2_3x3 = googlenet.add_convolution(l_conv2_reduce_relu, counter = 'conv2/3x3', num_output = 192, pad = 1, kernel_size = 3, **conv_kwargs)
l_conv2_3x3_relu = googlenet.add_relu(l_conv2_3x3, counter = 'conv2/relu_3x3')

l_lrn2 = googlenet.add_lrn(l_conv2_3x3_relu, counter = 'conv2/norm2', local_size = 5, alpha = 0.0001, beta = 0.75)
l_pool2 = googlenet.add_pooling(l_lrn2, counter = 'pool2/3x3_s2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)

l_inception5_3a = googlenet.add_inception_5(l_pool2, counter = 'inception_3a', **inception3a_kwargs)
l_inception5_3b = googlenet.add_inception_5(l_inception5_3a, counter = 'inception_3b', **inception3b_kwargs)

l_pool3 = googlenet.add_pooling(l_inception5_3b, counter = 'pool3/3x3_s2', pool = params.Pooling.MAX, kernel_size = 3, stride = 2)

l_inception5_4a = googlenet.add_inception_5(l_pool3, counter = 'inception_4a', **inception4a_kwargs)


l_loss1_avepool = googlenet.add_pooling(l_inception5_4a, counter = 'loss1/ave_pool', pool = params.Pooling.AVE, kernel_size = 5, stride = 3)
l_loss1_conv = googlenet.add_convolution(l_loss1_avepool, counter = 'loss1/conv', num_output = 128, kernel_size = 1, **conv_kwargs)
l_loss1_relu_conv = googlenet.add_relu(l_loss1_conv, counter = 'loss1/relu_conv')

l_loss1_fc = googlenet.add_dense(l_loss1_relu_conv, counter = 'loss1/fc',
                              num_output = 1024,
                              param = common_params, weight_filler = fill_xavier,
                              bias_filler = dict(type='constant', value = 0.2))
l_loss1_relu_fc = googlenet.add_relu(l_loss1_fc, counter = 'loss1/relu_fc')
l_loss1_drop_fc = googlenet.add_dropout(l_loss1_relu_fc, counter = 'loss1/drop_fc', dropout_ratio = 0.7)

l_loss1_clf = googlenet.add_dense(l_loss1_drop_fc, counter = 'loss1/classifier',
                              num_output = 1000,
                              param = common_params, weight_filler = fill_xavier,
                              bias_filler = dict(type='constant', value = 0))
l_loss1_softmax = googlenet.add_softmax_with_loss([l_loss1_clf, googlenet.blob_label()], counter = 'loss1/loss', loss_weight = 0.3)

l_loss1_accuracy1 = googlenet.add_accuracy([l_loss1_clf, googlenet.blob_label()], counter = 'loss1/top-1',
                                            include = dict(phase = 1))
l_loss1_accuracy5 = googlenet.add_accuracy([l_loss1_clf, googlenet.blob_label()], counter = 'loss1/top-5',
                                            include = dict(phase = 1), accuracy_param=dict(top_k=5))

l_inception5_4b = googlenet.add_inception_5(l_inception5_4a, counter = 'inception_4b', **inception4b_kwargs)
l_inception5_4c = googlenet.add_inception_5(l_inception5_4b, counter = 'inception_4c', **inception4c_kwargs)
l_inception5_4d = googlenet.add_inception_5(l_inception5_4c, counter = 'inception_4d', **inception4d_kwargs)

l_loss2_avepool = googlenet.add_pooling(l_inception5_4d, counter = 'loss2/ave_pool', pool = params.Pooling.AVE, kernel_size = 5, stride = 3)
l_loss2_conv = googlenet.add_convolution(l_loss2_avepool, counter = 'loss2/conv', num_output = 128, kernel_size = 1, **conv_kwargs)
l_loss2_relu_conv = googlenet.add_relu(l_loss2_conv, counter = 'loss2/relu_conv')

l_loss2_fc = googlenet.add_dense(l_loss2_relu_conv, counter = 'loss2/fc',
                              num_output = 1024,
                              param = common_params, weight_filler = fill_xavier,
                              bias_filler = dict(type='constant', value = 0.2))
l_loss2_relu_fc = googlenet.add_relu(l_loss2_fc, counter = 'loss2/relu_fc')
l_loss2_drop_fc = googlenet.add_dropout(l_loss2_relu_fc, counter = 'loss2/drop_fc', dropout_ratio = 0.7)

l_loss2_clf = googlenet.add_dense(l_loss2_drop_fc, counter = 'loss2/classifier',
                              num_output = 1000,
                              param = common_params, weight_filler = fill_xavier,
                              bias_filler = dict(type='constant', value = 0))
l_loss2_softmax = googlenet.add_softmax_with_loss([l_loss2_clf, googlenet.blob_label()], counter = 'loss2/loss', loss_weight = 0.3)

l_loss2_accuracy1 = googlenet.add_accuracy([l_loss2_clf, googlenet.blob_label()], counter = 'loss2/top-1',
                                            include = dict(phase = 1))
l_loss2_accuracy5 = googlenet.add_accuracy([l_loss2_clf, googlenet.blob_label()], counter = 'loss2/top-5',
                                            include = dict(phase = 1), accuracy_param=dict(top_k=5))

#}
#layer {
#  name: "loss2/ave_pool"
#  type: "Pooling"
#  bottom: "inception_4d/output"
#  top: "loss2/ave_pool"
#  pooling_param {
#    pool: AVE
#    kernel_size: 5
#    stride: 3
#  }
#}
#layer {
#  name: "loss2/conv"
#  type: "Convolution"
#  bottom: "loss2/ave_pool"
#  top: "loss2/conv"
#  param {
#    lr_mult: 1
#    decay_mult: 1
#  }
#  param {
#    lr_mult: 2
#    decay_mult: 0
#  }
#  convolution_param {
#    num_output: 128
#    kernel_size: 1
#    weight_filler {
#      type: "xavier"
#    }
#    bias_filler {
#      type: "constant"
#      value: 0.2
#    }
#  }
#}
#layer {
#  name: "loss2/relu_conv"
#  type: "ReLU"
#  bottom: "loss2/conv"
#  top: "loss2/conv"
#}
#layer {
#  name: "loss2/fc"
#  type: "InnerProduct"
#  bottom: "loss2/conv"
#  top: "loss2/fc"
#  param {
#    lr_mult: 1
#    decay_mult: 1
#  }
#  param {
#    lr_mult: 2
#    decay_mult: 0
#  }
#  inner_product_param {
#    num_output: 1024
#    weight_filler {
#      type: "xavier"
#    }
#    bias_filler {
#      type: "constant"
#      value: 0.2
#    }
#  }
#}
#layer {
#  name: "loss2/relu_fc"
#  type: "ReLU"
#  bottom: "loss2/fc"
#  top: "loss2/fc"
#}
#layer {
#  name: "loss2/drop_fc"
#  type: "Dropout"
#  bottom: "loss2/fc"
#  top: "loss2/fc"
#  dropout_param {
#    dropout_ratio: 0.7
#  }
#}
#layer {
#  name: "loss2/classifier"
#  type: "InnerProduct"
#  bottom: "loss2/fc"
#  top: "loss2/classifier"
#  param {
#    lr_mult: 1
#    decay_mult: 1
#  }
#  param {
#    lr_mult: 2
#    decay_mult: 0
#  }
#  inner_product_param {
#    num_output: 1000
#    weight_filler {
#      type: "xavier"
#    }
#    bias_filler {
#      type: "constant"
#      value: 0
#    }
#  }
#}
#layer {
#  name: "loss2/loss"
#  type: "SoftmaxWithLoss"
#  bottom: "loss2/classifier"
#  bottom: "label"
#  top: "loss2/loss1"
#  loss_weight: 0.3
#}
#layer {
#  name: "loss2/top-1"
#  type: "Accuracy"
#  bottom: "loss2/classifier"
#  bottom: "label"
#  top: "loss2/top-1"
#  include {
#    phase: TEST
#  }
#}
#layer {
#  name: "loss2/top-5"
#  type: "Accuracy"
#  bottom: "loss2/classifier"
#  bottom: "label"
#  top: "loss2/top-5"
#  include {
#    phase: TEST
#  }
#  accuracy_param {
#    top_k: 5
#  }
#}


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

l_softmax = googlenet.add_softmax_with_loss([l_dense1, googlenet.blob_label()], counter = 'loss3/loss3', loss_weight = 1)

l_loss1_accuracy1 = googlenet.add_accuracy([l_dense1, googlenet.blob_label()], counter = 'loss3/top-1',
                                            include = dict(phase = 1))
l_loss1_accuracy5 = googlenet.add_accuracy([l_dense1, googlenet.blob_label()], counter = 'loss3/top-5',
                                            include = dict(phase = 1), accuracy_param=dict(top_k=5))

print(str(googlenet))

text_file = open("googlenet.prototxt", "w")
text_file.write('name: \"GoogleNet\"\n')
text_file.write(str(googlenet))
text_file.close()