# Caffe Configuration Helpers

Wrapper around caffe.pycaffe for less-error-prone writing of network configurations

Usage example:

```Python
net = network.Network(batch_size = 32)
l_conv1 = net.add_convolution(net.blob_data(),
                              num_output = 64, kernel_size = 3, stride = 1,
                              param = dict(lr_mult = 1, decay_mult = 1))
l_relu1 = net.add_relu(l_pool1)
l_drop1 = net.add_dropout(l_conv1, dropout_ratio = 0.5)
l_conv2 = net.add_convolution(l_drop1, num_output = 32, weight_filler=dict(type='Xavier'))
l_pool2 = net.add_pooling(l_conv2, pool=params.Pooling.MAX)
l_relu2 = net.add_relu(l_pool2)
l_softmax = net.add_softmax_with_loss(l_relu2, loss_weight = 1)

print(str(net))
```

Result:

```Java
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  top: "label"
  transform_param {
    scale: 0.00392156862745
  }
  dummy_data_param {
    shape {
      dim: 32
      dim: 1
      dim: 28
      dim: 28
    }
    shape {
      dim: 32
      dim: 1
      dim: 1
      dim: 1
    }
  }
}
layer {
  name: "conv11"
  type: "Convolution"
  bottom: "data"
  top: "conv11"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
  }
}
layer {
  name: "DummyData1"
  type: "DummyData"
  top: "DummyData1"
  top: "DummyData2"
  transform_param {
    scale: 0.00392156862745
  }
  dummy_data_param {
    shape {
      dim: 32
      dim: 1
      dim: 28
      dim: 28
    }
    shape {
      dim: 32
      dim: 1
      dim: 1
      dim: 1
    }
  }
}
layer {
  name: "Convolution1"
  type: "Convolution"
  bottom: "DummyData1"
  top: "Convolution1"
}
layer {
  name: "Dropout1"
  type: "Dropout"
  bottom: "Convolution1"
  top: "Dropout1"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "Convolution2"
  type: "Convolution"
  bottom: "Dropout1"
  top: "Convolution2"
  convolution_param {
    num_output: 32
    weight_filler {
      type: "Xavier"
    }
  }
}
layer {
  name: "Pooling1"
  type: "Pooling"
  bottom: "Convolution2"
  top: "Pooling1"
  pooling_param {
    pool: MAX
  }
}
layer {
  name: "relu12"
  type: "ReLU"
  bottom: "Pooling1"
  top: "Pooling1"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "conv11"
  top: "drop6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "conv12"
  type: "Convolution"
  bottom: "drop6"
  top: "conv12"
  convolution_param {
    num_output: 32
    weight_filler {
      type: "Xavier"
    }
  }
}
layer {
  name: "pool6"
  type: "Pooling"
  bottom: "conv12"
  top: "pool6"
  pooling_param {
    pool: MAX
  }
}
layer {
  name: "relu13"
  type: "ReLU"
  bottom: "pool6"
  top: "pool6"
}
layer {
  name: "classifier5"
  type: "SoftmaxWithLoss"
  bottom: "pool6"
  top: "classifier5"
  loss_weight: 1
}
```
