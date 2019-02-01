import os
import sys
import cv2
import time
import logging
import json
import tensorflow as tf
import numpy as np
import glob
import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import src
from src.data_manager import EmojifierDataManager
from src.__init__ import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

logger = logging.getLogger('emojifier.model')


def weight_variable(shape):
    initial = tf.truncated_normal(
        shape=shape, stddev=0.1
    )
    
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1,1,1,1], padding='SAME'
    )


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'
    )


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])

    return tf.nn.relu(tf.layers.batch_normalization(conv2d(input, W) + b))


def full_layer(input, size):
    insize = int(input.get_shape()[1])
    W = weight_variable([insize, size])
    b = bias_variable([size])

    return tf.matmul(input, W) + b


def model(x, keep_prob):

    C1, C2, C3 = 30, 50, 80
    F1 = 512
    conv1_1 = conv_layer(x, shape=[3, 3, 1, C1])
    conv1_1_pool = max_pool_2x2(conv1_1)
    
    conv1_2 = conv_layer(conv1_1_pool, shape=[3, 3, C1, C2])
    conv1_2_pool = max_pool_2x2(conv1_2)
    
    conv1_drop = tf.nn.dropout(conv1_2_pool, keep_prob=keep_prob) 
    
    conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C2, C3])
    conv2_1_pool = max_pool_2x2(conv2_1)

    conv2_flat = tf.reshape(conv2_1_pool, [-1, 6*6*C3])
    conv2_drop = tf.nn.dropout(conv2_flat, keep_prob=keep_prob)

    full1 = tf.nn.relu(full_layer(conv2_drop, F1))
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)
    
    y_conv = full_layer(full1_drop, len(EMOTION_MAP))

    return y_conv


def test(emoji_data, sess):
    logger.info('CALCULATING TESTSET ACCURACY ...')
    L = len(emoji_data.test.labels)

    x = emoji_data.test.images
    y = emoji_data.test.labels

    accs = []

    for i in tqdm.tqdm(range(0, L, 30)):
        if i+30 <= L:
            x_i = x[i:i+30].reshape(30, 48, 48, 1)
            y_i = y[i:i+30].reshape(30, len(EMOTION_MAP))
        else:
            x_i = x[i:].reshape(L-i, 48, 48, 1)
            y_i = y[i:].reshape(L-i, len(EMOTION_MAP))

        accs.append(sess.run(accuracy, feed_dict={X:x_i, Y:y_i, keep_prob:1.0}))
        
    acc = np.mean(accs)
    
    logger.critical('test-accuracy: {:.4}%'.format(acc*100))


if __name__ == '__main__':

    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'model_checkpoints')

    if not os.path.exists(CHECKPOINT_SAVE_PATH):
        os.makedirs(CHECKPOINT_SAVE_PATH)

    BATCH_SIZE = config_parser.getint('MODEL_HYPER_PARAMETERS', 'batch_size')
    STEPS = config_parser.getint('MODEL_HYPER_PARAMETERS', 'train_steps')
    LEARNING_RATE = config_parser.getfloat('MODEL_HYPER_PARAMETERS', 'learning_rate')
    KEEP_PROB = config_parser.getfloat('MODEL_HYPER_PARAMETERS', 'dropout_keep_prob')

    X = tf.placeholder(
        tf.float32, shape=[None, 48, 48, 1]
    )
    
    Y = tf.placeholder(tf.float32, shape=[None, len(EMOTION_MAP)])
    
    keep_prob = tf.placeholder(tf.float32)

    emoset = EmojifierDataManager()

    logger.info("Number of train images: {}".format(
        len(emoset.train.images)
    ))
    logger.info("Number of train labels: {}".format(
        len(emoset.train.labels)
    ))
    logger.info("Number of test images: {}".format(
        len(emoset.test.images)
    ))
    logger.info("Number of test labels: {}".format(
        len(emoset.test.labels)
    ))

    y_conv = model(X, keep_prob)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=Y,
            logits=y_conv
        )
    )
    
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_predictions = tf.equal(
        tf.argmax(y_conv, 1), tf.argmax(Y, 1)
    )
    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32)
    )
    
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())

        for i in tqdm.tqdm(range(STEPS)):
            x_data, y_data = emoset.train.next_batch(BATCH_SIZE)

            acc, loss, _ = sess.run(
                [accuracy, cross_entropy, train], 
                feed_dict={X:x_data, Y:y_data, keep_prob: KEEP_PROB}
            )

            if i % 20 == 0:
                logger.info('accuracy: {:.4}%, loss: {:.4}'.format(
                    acc*100, loss
                ))

        test(emoset, sess)
        
        save_path = saver.save(sess, os.path.join(CHECKPOINT_SAVE_PATH, 'model.ckpt'))
        
        logger.info("Model saved in path: {}".format(save_path))
    