#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2018/10/25 11:49
# @Author     : Li Shanlu
# @File       : preprocessing_data.py
# @Software   : PyCharm
# @Description: read the data and generate batches

import tensorflow as tf
import numpy as np
import os
import random


def get_img_path_and_lab(data_dir, split=True, split_val_ratio=0.2, shuffle=True):
    """
    :param data_dir: The directory include cat and dog images
    :param split: Whether or not split the data set
    :param split_val_ratio: If split is ture, the val set ratio
    :param shuffle: Whether or not shuffle the data set
    :return: list of image path and corresponding label
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for img_name in os.listdir(data_dir):
        img_path = data_dir + '/' + img_name
        if 'cat' in img_name:
            cats.append(img_path)
            label_cats.append(0)
        else:
            dogs.append(img_path)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))
    #切分数据库
    if split:
        n_cat = len(cats)
        n_dog = len(dogs)
        train_cats = cats[0:int(round((1-split_val_ratio)*n_cat))]
        train_cats_label = label_cats[0:int(round((1-split_val_ratio)*n_cat))]
        train_dogs = dogs[0:int(round((1-split_val_ratio)*n_dog))]
        train_dogs_label = label_dogs[0:int(round((1-split_val_ratio)*n_dog))]

        val_cats = cats[int(round((1-split_val_ratio)*n_cat)):]
        val_cats_label = label_cats[int(round((1-split_val_ratio)*n_cat)):]
        val_dogs = dogs[int(round((1-split_val_ratio)*n_dog)):]
        val_dogs_label = label_dogs[int(round((1-split_val_ratio)*n_dog)):]

        train_list = np.hstack((train_cats, train_dogs))
        train_label = np.hstack((train_cats_label, train_dogs_label))
        val_list = np.hstack((val_cats, val_dogs))
        val_label = np.hstack((val_cats_label, val_dogs_label))
    else:
        train_list = np.hstack((cats, dogs))
        train_label = np.hstack((label_cats, label_dogs))
        val_list = []
        val_label = []
    # 打乱数据
    if shuffle:
        shuffle_train_list = list(zip(train_list, train_label))
        random.shuffle(shuffle_train_list)
        train_list, train_label = zip(*shuffle_train_list)
        if len(val_list):
            shuffle_val_list = list(zip(val_list, val_label))
            random.shuffle(shuffle_val_list)
            val_list, val_label = zip(*shuffle_val_list)

    return train_list, train_label, val_list, val_label


#获取批量形式的数据
def get_batch_data(image, label, image_w, image_h, batch_size, capacity):
    """
    :param image: 要生成batch的图像
    :param label: 要生成batch的标签
    :param image_w: 生成batch后图片的宽
    :param image_h: 生成batch后图片的高
    :param batch_size: 每个batch包含的图片张数
    :param capacity: 队列容量
    :return: 图像和标签的batch
    """
    #将list转换成tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    #生成队列
    input_queue = tf.train.slice_input_producer([image, label])
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)
    #统一图片大小
    #image = tf.image.resize_image_with_crop_or_pad(image, image_h, image_w)  #原图比image_h和image_w小，则pad，否则为crop
    image = tf.image.resize_images(image, size=[image_h, image_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.per_image_standardization(image)      #标准化数据，即减均值，除以方差
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=4,
                                              capacity=capacity)
    # You can also use shuffle_batch
    """
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=4,
                                                      capacity=capacity,
                                                      min_after_dequeue=capacity-1)
    """
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


def parse_fn(filename, label):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_image(file_contents, channels=3)
    # add by lishanlu, for train arcface with imagesize=112*112
    # image = tf.image.decode_jpeg(file_contents, channels=3)
    # image = tf.image.resize_images(image, (128, 128))
    image = tf.random_crop(image, [160, 160, 3])
    image = tf.image.random_flip_left_right(image)

    image = tf.image.per_image_standardization(image)

    image.set_shape((160, 160, 3))
    label.set_shape(())

    return image, label

#演示效果
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    batch_size = 2
    capacity = 256
    img_w = 160
    img_h = 160
    data_dir = "D:/Dataset/kaggle_cat vs dog/train"
    train_list, train_label, val_list, val_label = get_img_path_and_lab(data_dir)
    print("Train image: %d" %(len(train_list)))
    print("Val image: %d" %(len(val_list)))
    image_batch, label_batch = get_batch_data(train_list, train_label, img_w, img_h, batch_size, capacity)
    """    
    train_list = tf.cast(train_list, tf.string)
    train_label = tf.cast(train_label, tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((train_list, train_label))
    tf.data.Dataset.shuffle(dataset, buffer_size=10000, reshuffle_each_iteration=1000)
    dataset = tf.data.Dataset.repeat(10)
    #dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, 10))
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(2000)
    iterator = dataset.make_one_shot_iterator()
    #iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()
    """
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 5:

                img, label = sess.run([image_batch, label_batch])

                for j in np.arange(batch_size):
                    print("label: %d" %label[j])
                    import cv2
                    hsv = cv2.cvtColor(img[j,:,:,:], cv2.COLOR_BGR2HSV)
                    img_hsv = np.asarray(hsv, dtype='uint8')
                    plt.imshow(img_hsv)
                    plt.show()
                    #img_j = np.asarray(img[j,:,:,:], dtype='uint8')
                    #plt.imshow(img_j)
                    #plt.show()

                i += 1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)




