import numpy as np
import tensorflow as tf
import scipy.io
import os
from six.moves import urllib


class VGG:
    def __init__(self, input_img):
        self.vgg_link = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
        self.vgg_file_path = 'imagenet-vgg-verydeep-19.mat'  # dictionary containing the CNN structure
        self.input_img = input_img

    def donwload_vgg(self):
        if os.path.exists(self.vgg_file_path):
            return
        urllib.request.urlretrieve(self.vgg_link, self.vgg_file_path)

    def extract_data(self, layer_name, layer_index):
        mat = scipy.io.loadmat(self.vgg_file_path)
        weights = mat['layers'][0][layer_index][0][0][2][0][0]
        bias = mat['layers'][0][layer_index][0][0][2][0][1] #bias.shape = (64,1), size = 64
        name = mat['layers'][0][layer_index][0][0][0][0]
        assert name == layer_name
        return weights, bias.reshape(bias.size)

    def conv_relu(self, input_layer, layer_name, layer_index, scope_name='conv'):
        with tf.name_scope(scope_name) as scope:
            W_, b_ = self.extract_data(layer_name, layer_index)
            weight = tf.get_variable('weight', initializer=W_)
            bias = tf.get_variable('bias', initializer=b_)
            conv = tf.layers.conv2d(input_layer, weight, strides=[1,1,1,1], padding= 'SAME')
            out = tf.nn.relu(conv+bias)
            return out

    def avgpool(self, input_layer, scope_name='pool'):
        with tf.name_scope(scope_name) as scope:
            pool = tf.layers.average_pooling2d(input_layer, strides=[1,2,2,1], padding='SAME')
            return pool

    def build(self):
        self.conv1_1 = self.conv_relu(self.input_img, 'conv1_1',0,'conv1_1')
        self.conv1_2 = self.conv_relu(self.conv1_1, 'conv1_2', 2, 'conv1_2')
        self.pool1 = self.avgpool(self.conv1_2, 'pool1')
        self.conv2_1 = self.conv_relu(self.pool1, 'conv2_1', 5, 'conv2_1')
        self.conv2_2 = self.conv_relu(self.conv2_1, 'conv2_2', 7, 'conv2_2')
        self.pool2 = self.avgpool(self.conv2_2, 'pool2')
        self.conv3_1 = self.conv_relu(self.pool2, 'conv3_1', 10, 'conv3_1')
        self.conv3_2 = self.conv_relu(self.conv3_1, 'conv3_2',11, 'conv3_2')
        self.conv3_3 = self.conv_relu(self.conv3_2, 'conv3_3', 12, 'conv3_3')
        self.conv3_4 = self.conv_relu(self.conv3_3, 'conv3_4', 13, 'conv3_4')
        self.pool3 = self.avgpool(self.conv3_4, 'pool3')
        self.conv4_1 = self.conv_relu(self.pool3, 'conv4_1', 16, 'conv4_1')
        self.conv4_2 = self.conv_relu(self.conv4_1, 'conv4_2', 17, 'conv4_2')
        self.conv4_3 = self.conv_relu(self.conv4_2, 'conv4_3', 18, 'conv4_3')
        self.conv4_4 = self.conv_relu(self.conv4_3, 'conv4_4', 19, 'conv4_4')
        self.pool4 = self.avgpool(self.conv4_4, 'pool4')
        self.conv5_1 = self.conv_relu(self.pool4, 'conv5_1', 22, 'conv5_1')
        self.conv5_2 = self.conv_relu(self.conv5_1,'conv5_2', 23, 'conv5_2')
        self.conv5_3 = self.conv_relu(self.conv5_2, 'conv5_3', 24, 'conv5_3')
        self.conv5_4 = self.conv_relu(self.conv5_3, 'conv5_4', 25, 'conv5_4')
        self.pool5 = self.avgpool(self.conv5_4, 'pool5')


if __name__ == '__main__':
    mat = VGG()
    mat.build()
