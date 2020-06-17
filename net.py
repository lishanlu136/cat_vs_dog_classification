#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2018/10/25 11:48
# @Author     : Li Shanlu
# @File       : net.py
# @Software   : PyCharm
# @Description: 定义分类网络


import tensorflow as tf
import tensorflow.contrib.slim as slim


def inference(images, num_classes, dropout_rate=0.8, is_training=True, weight_decay=4e-4, scope="My_Net"):
    """
    :param images: Input images, tensor for[batch_size, x, x, 3]
    :param num_classes: number of image class
    :param dropout_rate: rate for dropout
    :param is_training: Whether or not training
    :param weight_decay: regularing args for weight
    :return: logits, endpoints of the defined network
    """
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    with tf.variable_scope(scope, [images]):
        end_point = {}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.xavier_initializer_conv2d(),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                net = slim.conv2d(images, 64, [3, 3], stride=2, scope='conv1')
                end_point['conv1'] = net
                net = slim.conv2d(net, 64, [3, 3], stride=2, scope='conv2')
                end_point['conv2'] = net
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool2')
                end_point['maxpool2'] = net
                net = slim.conv2d(net, 96, [3, 3], scope='conv3')
                end_point['conv3'] = net
                net = slim.conv2d(net, 96, [3, 3], scope='conv4')
                end_point['conv4'] = net
                net = slim.conv2d(net, 96, [3, 3], scope='conv5')
                end_point['conv5'] = net
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool5')
                end_point['maxpool5'] = net
                net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv6')
                end_point['conv6'] = net
                net = slim.dropout(net, dropout_rate)
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv7')
                end_point['conv7'] = net
                net = slim.avg_pool2d(net, kernel_size=net.get_shape()[1:-1], stride=1, scope='global_avg_pool7')
                end_point['avgpool7'] = net
                net = tf.squeeze(net, [1, 2], name='logits')
                end_point['out'] = net
    return net, end_point


def loss(logits, labels):
    #labels = tf.cast(labels, tf.int64)
    # 注意，这里上面定义的不是ont-hot编码，故这里调用的是sparse方法
    # 如果使用的是ont-hot编码，则需调用softmax_cross_entropy_with_logits即可
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits, name='cross_entropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return loss


def accuracy(logits, labels, num_classes):
    # 将labels转换为ont-hot编码计算
    #labels = tf.one_hot(labels, num_classes)
    #correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_pred = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


#测试网络的正确性
if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 160, 160, 3), name='input')
        logits, end_points = inference(inputs, 2)
        print("Layers:")
        for k, v in end_points.items():
            print('name = {}, shape = {}'.format(v.name, v.get_shape()))
        print('\n')
        print("Parameters:")
        for v in slim.get_model_variables():
            print('name={}, shape={}'.format(v.name, v.get_shape()))





