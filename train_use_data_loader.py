#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/3/12 10:10
@Author     : Li Shanlu
@File       : train_use_data_loader.py
@Software   : PyCharm
@Description:  train net use data loader api
"""
import preprocessing_data
from data_loader import DataLoader
import net
import tensorflow as tf
from tensorflow.contrib import slim
import time
import numpy as np
import argparse
import os
import sys
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    if args.val_data_dir:
        train_list, train_label, _, _ = preprocessing_data.get_img_path_and_lab(args.train_data_dir, split=False, shuffle=True)
        val_list, val_label, _, _ = preprocessing_data.get_img_path_and_lab(args.val_data_dir, split=False, shuffle=True)
    else:
        train_list, train_label, val_list, val_label = preprocessing_data.get_img_path_and_lab(args.train_data_dir)

    subdir = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    log_dir = os.path.join(os.path.expanduser(args.logs_dir), subdir)
    if os.path.exists(log_dir):
        os.rmdir(log_dir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_dir), subdir)
    if os.path.exists(model_dir):
        os.rmdir(model_dir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    train_dataset = DataLoader(train_list, train_label, [160, 160], num_classes=2)
    val_dataset = DataLoader(val_list, val_label, [160, 160], num_classes=2)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        input_placeholder = tf.placeholder(tf.float32, [None, 160, 160, 3], name="input")
        label_placeholder = tf.placeholder(tf.int64, [None, ], name="label")
        #keep_prob_placeholder = tf.placeholder(tf.float32, name="dropout_prob")
        #phase_train_placeholder = tf.placeholder(tf.bool, name="phase_train")

        logits, end_points = net.inference(input_placeholder, num_classes=2, dropout_rate=args.dropout_rate,
                                           is_training=True, weight_decay=args.weight_decay, scope="My_Net")
        # Loss
        cross_entropy_mean = net.loss(logits, label_placeholder)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = cross_entropy_mean + tf.add_n(regularization_loss)
        # Prediction
        prob = tf.nn.softmax(logits=logits, name='prob')
        # Accuracy
        accuracy_op = net.accuracy(logits, label_placeholder, 2)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        train_op = optimizer.minimize(loss=total_loss, global_step=global_step)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        train_data, train_init = train_dataset.data_batch(augment=True, shuffle=True, batch_size=32, repeat_times=1000, num_threads=4, buffer=5000)
        val_data, val_init = val_dataset.data_batch(augment=True, shuffle=False, batch_size=32, repeat_times=1, num_threads=4, buffer=5000)

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_init)
        sess.run(val_init)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        max_accuracy = 0
        with sess.as_default():
            print("Running training.")
            num_step = args.epoch_size * args.max_nrof_epochs
            for step in range(1, num_step + 1):
                train_element = sess.run(train_data)
                _, = sess.run([train_op],feed_dict={input_placeholder: train_element[0],label_placeholder: train_element[1]})
                if (step % args.val_step) == 0 or step == 1:
                    # Calculate Validation loss and accuracy
                    val_element = sess.run(val_data)
                    loss, acc = sess.run([total_loss, accuracy_op], feed_dict={input_placeholder: val_element[0],
                                                                               label_placeholder: val_element[1]})
                    if acc > max_accuracy:
                        save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, global_step)

                    print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                        loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str,
                        help='Directory where the train images.', default='D:/Dataset/public_dataset/kaggle_cat vs dog/train')
    parser.add_argument('--val_data_dir', type=str,
                        help='Directory where the validation images.', default='')
    parser.add_argument('--logs_dir', type=str,
                        help='Directory where to write event logs.', default='train_results/logs')
    parser.add_argument('--models_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='train_results/models')
    parser.add_argument('--gpu_idx', type=str,
                        help='gpu indexs', default='0')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9)
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=100)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=50)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=32)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--dropout_rate', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=2e-4)
    parser.add_argument('--learning_rate', type=float,
                        help='learning rate for train.', default=0.001)
    parser.add_argument('--val_step', type=int,
                        help='validation after how many training steps.', default=100)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
