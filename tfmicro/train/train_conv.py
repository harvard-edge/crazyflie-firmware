"""
Simple training script that maps 4 possible values from 0-2000
(information from our distance sensor at each axis)
to a simple policy of going in a direction. 

INPUTS:
    We will preprocess the values 0-2000 to integer values between
    0-200 by dividing by 10. These will then be the distance in
    centimeters.

    The inputs would be mapped:
        0 -> front
        1 -> right
        2 -> back
        3 -> left

OUTPUTS:
    We will map to 9 possible values
        0 -> do nothing, hover
        1 -> forward
        2 -> forward-right
        3 -> right
        ... (proceeding in clockwise order)
        7 -> left
        8 -> forward-left
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf

import models
from utils import generate_simulated_data
from tensorflow.python.platform import gfile
import keras.utils
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

FLAGS = None


def main(_):
    NUM_INPUTS = FLAGS.input_size
    NUM_HISTORY = FLAGS.history_size  # amount of history we are keeping of previous measurements
    NUM_CLASSES = FLAGS.output_size
    
    # the data, split between train and test sets
    x_train, y_train, x_test, y_test = generate_simulated_data(10000, balance=False)
    # append more data to x_train and x_test
    for _ in range(NUM_HISTORY - 1):
        a, _, at, _ = generate_simulated_data(10000, balance=False)
        x_train = np.concatenate([x_train, a], axis=1) 
        x_test = np.concatenate([x_test, at], axis=1) 


    x_train = x_train.astype('uint8')
    x_test = x_test.astype('uint8')
    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    # convert class vectors to binary class matrices
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
            len(learning_rates_list)))

    input_placeholder = tf.placeholder(tf.float32, [None, NUM_INPUTS * NUM_HISTORY], name='graph_input')
    if FLAGS.quantize:
        input_min, input_max = 0, 256 
        graph_input = tf.fake_quant_with_min_max_args(input_placeholder, input_min, input_max)
    else:
        graph_input = input_placeholder

    logits, dropout_prob = models.create_conv_model(
        graph_input, NUM_INPUTS, NUM_HISTORY, NUM_CLASSES, is_training=True)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(tf.int64, [None], name='groundtruth_input')

    # Optionally we can add runtime checks to spot when NaNs or other symptoms of
    # numerical errors start occurring during training.
    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_input, logits=logits)
    if FLAGS.quantize:
        tf.contrib.quantize.create_training_graph(quant_delay=0)
    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
        train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, ground_truth_input)
    confusion_matrix = tf.confusion_matrix(
            ground_truth_input, predicted_indices, num_classes=NUM_CLASSES)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.get_default_graph().name_scope('eval'):
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        tf.summary.scalar('accuracy', evaluation_step)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all(scope='eval')
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1
    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir, FLAGS.model_architecture + '.pbtxt')

    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    for training_step in xrange(start_step, training_steps_max + 1):
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break

        # Pull the audio samples we'll use for training.
        index = (training_step * FLAGS.batch_size) % x_train.shape[0]
        train_fingerprints = x_train[index:index+FLAGS.batch_size]
        train_ground_truth = y_train[index:index+FLAGS.batch_size]
        # Run the graph with this batch of training data.
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries,
                evaluation_step,
                cross_entropy_mean,
                train_step,
                increment_global_step,
            ],
            feed_dict={
                graph_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
                dropout_prob: 0.5
            })
        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                            cross_entropy_value))

        is_last_step = (training_step == training_steps_max)
        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            set_size = y_test.shape[0] 
            total_accuracy = 0
            total_conf_matrix = None
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprints = x_test[i:i+FLAGS.batch_size]
                validation_ground_truth = y_test[i:i+FLAGS.batch_size]
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, confusion_matrix],
                    feed_dict={
                        graph_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
                        dropout_prob: 1.0
                    })
                validation_writer.add_summary(validation_summary, training_step)
                batch_size = min(FLAGS.batch_size, set_size - i)
                total_accuracy += (validation_accuracy * batch_size) / set_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix
            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                            (training_step, total_accuracy * 100, set_size))

        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or training_step == training_steps_max):
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_size',
        dest='input_size',
        default=4,
        type=int,
        help='first dimension to input of conv net')
    parser.add_argument(
        '--output_size',
        dest='output_size',
        default=9,
        type=int,
        help='output dimension of conv net')
    parser.add_argument(
        '--history_size',
        dest='history_size',
        default=10,
        type=int,
        help='second dimension to input of conv net, historical measurements')
    parser.add_argument(
            '--how_many_training_steps',
            type=str,
            default='4000,4000,4000',
            help='How many training loops to run',)
    parser.add_argument(
            '--eval_step_interval',
            type=int,
            default=5000,
            help='How often to evaluate the training results.')
    parser.add_argument(
            '--learning_rate',
            type=str,
            default='0.001,0.0001,0.00005',
            help='How large a learning rate to use when training.')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=2048,
            help='How many items to train with at once',)
    parser.add_argument(
            '--summaries_dir',
            type=str,
            default='/tmp/retrain_logs',
            help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
            '--train_dir',
            type=str,
            default='/tmp/tfmicro/conv',
            help='Directory to write event logs and checkpoint.')
    parser.add_argument(
            '--save_step_interval',
            type=int,
            default=100,
            help='Save model checkpoint every save_steps.')
    parser.add_argument(
            '--model_architecture',
            type=str,
            default='conv',
            help='What model architecture to use')
    parser.add_argument(
            '--check_nans',
            type=bool,
            default=False,
            help='Whether to check for invalid numbers during processing')
    parser.add_argument(
            '--quantize',
            type=bool,
            default=True,
            help='Whether to train the model for eight-bit deployment')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
