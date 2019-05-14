"""Converts a trained checkpoint into a frozen model for microcontroller
inference. This script is primarily for fullyconnected or conv models that we
will quantize and deploy to the Crazyflie.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
import models
from tensorflow.python.framework import graph_util

FLAGS = None

def create_inference_graph():
    """Creates an model with the nodes needed for inference."""
    fingerprint_size = FLAGS.num_inputs 
    reshaped_input = tf.placeholder(tf.float32, [None, fingerprint_size], name="input")
    logits = models.create_two_fc_model(
        reshaped_input,
        FLAGS.num_inputs,
        FLAGS.fc_size,
        FLAGS.num_outputs,
        is_training=False)
    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')


def main(_):
    # Create the model and load its weights.
    sess = tf.InteractiveSession()
    create_inference_graph()
    if FLAGS.quantize:
        tf.contrib.quantize.create_eval_graph()
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['labels_softmax'])
    tf.train.write_graph(
            frozen_graph_def,
            os.path.dirname(FLAGS.output_file),
            os.path.basename(FLAGS.output_file),
            as_text=False)
    tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--output_file', type=str, help='Where to save the frozen graph.')
    parser.add_argument(
        '--num_inputs', type=int, help='Dimensionality of 1D input',
        default=4)
    parser.add_argument(
        '--fc_size', type=int, dest='fc_size', help='fullyconnected size',
        default=80)
    parser.add_argument(
        '--num_outputs', type=int, help='Dimensionality of output',
        default=9)
    parser.add_argument(
        '--quantize',
        type=bool,
        default=True,
        help='Whether to train the model for eight-bit deployment')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
