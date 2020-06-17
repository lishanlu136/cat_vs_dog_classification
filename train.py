#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2018/10/25 11:50
# @Author     : Li Shanlu
# @File       : train.py
# @Software   : PyCharm
# @Description: 训练分类网络


import preprocessing_data
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


    """
    image_batch, label_batch = preprocessing_data.get_batch_data(train_list, train_label,
                                                                 args.image_size, args.image_size,
                                                                 args.batch_size, 256)
    logits, end_points = net.inference(image_batch, num_classes=2, dropout_rate=args.dropout_rate,
                                       is_training=True, weight_decay=args.weight_decay, scope="My_Net")
    # Loss
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                       labels=label_batch,
                                                                                       ), name="cross_entropy_mean")
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_mean + tf.add_n(regularization_loss)
    # Prediction
    prob = tf.nn.softmax(logits=logits, name='prob')
    # Accuracy
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
    accuracy_op = tf.reduce_mean(correct_prediction)
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss=total_loss, global_step=global_step)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(1000):
            print(step)
            if coord.should_stop():
                break
            _, train_acc, train_loss = sess.run([train_op, accuracy_op, total_loss])
            print("loss:{} accuracy:{}".format(train_loss, train_acc))
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    """
    # 构建图
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        assert len(train_list)>0, 'The training set should not be empty.'

        # Create a queue that produces indices into the image_list and label_list
        labels = tf.convert_to_tensor(train_label, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None, shuffle=True)
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=200000,
                                              dtypes=[tf.string, tf.int32],
                                              shapes=[(1,), (1,)])
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                #image = tf.image.resize_images(image, size=[args.image_size, args.image_size],
                #                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                image = tf.image.per_image_standardization(image)       # 标准化数据，即减均值，除以方差
                images.append(image)
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size_placeholder,
                                                       shapes=[(args.image_size, args.image_size, 3), ()],
                                                       enqueue_many=True, capacity=4 * nrof_preprocess_threads * args.batch_size,
                                                       allow_smaller_final_batch=True)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print("Building training graph.")
        logits, end_points = net.inference(image_batch, num_classes=2, dropout_rate=args.dropout_rate,
                                           is_training=True, weight_decay=args.weight_decay, scope="My_Net")
        # Loss
        cross_entropy_mean = net.loss(logits, label_batch)
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = cross_entropy_mean + tf.add_n(regularization_loss)
        # Prediction
        prob = tf.nn.softmax(logits=logits, name='prob')
        # Accuracy
        accuracy_op = net.accuracy(logits, label_batch, 2)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        train_op = optimizer.minimize(loss=total_loss, global_step=global_step)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        with sess.as_default():
            print("Running training.")
            for epoch in range(args.max_nrof_epochs+1):
                if coord.should_stop():
                    break
                train(args, sess, epoch, train_list, train_label, index_dequeue_op, enqueue_op, image_paths_placeholder,
                      labels_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                      total_loss, train_op, summary_op, summary_writer, accuracy_op)
                # 每个epoch结束,保存模型
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, global_step)
                validate(args, sess, epoch, val_list, val_label, enqueue_op, image_paths_placeholder,
                         labels_placeholder,phase_train_placeholder, batch_size_placeholder, total_loss, accuracy_op)


def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder, phase_train_placeholder, batch_size_placeholder, step,
          loss, train_op, summary_op, summary_writer, accuracy):
    batch_number = 0

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {phase_train_placeholder: True, batch_size_placeholder: args.batch_size}
        tensor_list = [loss, train_op, step, accuracy]
        if batch_number % 100 == 0:
            loss_, _, step_, accuracy_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, accuracy_ = sess.run(tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        print(
            'Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f\t' %
            (epoch, batch_number + 1, args.epoch_size, duration, loss_, accuracy_, ))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True


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
    """
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables',
                      simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph',
                      simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
    """


def validate(args, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder,
             phase_train_placeholder, batch_size_placeholder, loss, accuracy):
    print('Running forward pass on validation set')

    nrof_batches = len(label_list) // args.batch_size
    nrof_images = nrof_batches * args.batch_size

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_list[:nrof_images]), 1)
    image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    loss_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: args.batch_size}
        loss_, accuracy_ = sess.run([loss, accuracy], feed_dict=feed_dict)
        loss_array[i], accuracy_array[i] = (loss_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time
    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f' %
          (epoch, duration, np.mean(loss_array), np.mean(accuracy_array)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str,
                        help='Directory where the train images.', default='D:/Dataset/kaggle_cat vs dog/train')
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

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




