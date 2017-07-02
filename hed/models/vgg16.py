# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os
import inspect

import numpy as np
from termcolor import colored
import tensorflow as tf

from hed.losses import sigmoid_cross_entropy_balanced
from hed.utils.io import IO


class Vgg16():

    def __init__(self, cfgs, run='training'):

        self.cfgs = cfgs
        self.io = IO()

        self.data_dict = np.load(self.cfgs['model_weights_path'], encoding='latin1').item()
        self.io.print_info("Model weights loaded from {}".format(self.cfgs['model_weights_path']))

        self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], self.cfgs[run]['n_channels']])
        self.edgemaps = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])

        self.define_model()

    def define_model(self):

        """
        Load VGG params from disk without FC layers A
        Add branch layers (with deconv) after each CONV block
        """

        start_time = time.time()

        self.conv1_1 = self.conv_layer(self.images, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.side_1 = self.deconv_layer(self.conv1_2, "side_1", 1)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.io.print_info('Added CONV-BLOCK-1+SIDE-1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.side_2 = self.deconv_layer(self.conv2_2, "side_2", 2)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.io.print_info('Added CONV-BLOCK-2+SIDE-2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.side_3 = self.deconv_layer(self.conv3_3, "side_3", 4)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.io.print_info('Added CONV-BLOCK-3+SIDE-3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.side_4 = self.deconv_layer(self.conv4_3, "side_4", 8)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.io.print_info('Added CONV-BLOCK-4+SIDE-4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.side_5 = self.deconv_layer(self.conv5_3, "side_5", 16)

        self.io.print_info('Added CONV-BLOCK-5+SIDE-5')

        self.side_outputs = [self.side_1, self.side_2, self.side_3, self.side_4, self.side_5]

        self.fuse = tf.layers.conv2d(tf.concat(self.side_outputs, axis=3), 1, 1, 1,
                                     padding='SAME', activation=tf.identity, use_bias=False,
                                     kernel_initializer=tf.constant_initializer(0.2))

        self.io.print_info('Added FUSE layer')

        self.outputs = self.side_outputs + [self.fuse]

        self.data_dict = None
        self.io.print_info("Build model finished: {:.4f}s".format(time.time() - start_time))

    def setup_testing(self, session):

        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            self.predictions.append(output)

    def setup_training(self, session):

        self.cost = []
        self.predictions = []
        self.loss = 0

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='xentropy{}'.format(idx + 1))

            self.cost.append(cost)
            self.predictions.append(output)
            self.loss += (self.cfgs['loss_weights'] * cost)

        predictions = tf.cast(tf.greater(output, self.cfgs['testing_threshold']), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(predictions, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('error', self.error)

        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/train', session.graph)
        self.test_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/test')

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def deconv_layer(self, inputs, name, upscale):

        with tf.variable_scope(name):

            reduction = tf.layers.conv2d(inputs, 1, 1, 1,
                                         padding='SAME', activation=tf.identity, use_bias=True,
                                         kernel_initializer=tf.constant_initializer(),
                                         bias_initializer=tf.constant_initializer())

            return tf.layers.conv2d_transpose(reduction, 1, [upscale * 2, upscale * 2],
                                              strides=[upscale, upscale], use_bias=False, padding='SAME',
                                              kernel_initializer=tf.random_uniform_initializer())

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
