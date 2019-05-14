# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.
This is from the tensorflow repository, and it can also be used for training
simple DQNs with the crazyflie.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


def create_micro_conv_model(fingerprint_input, height, width, output_shape, is_training):
    """Builds a convolutional model aimed at microcontrollers.

    Devices like DSPs and microcontrollers can have very small amounts of
    memory and limited processing power. This model is designed to use less
    than 20KB of working RAM, and fit within 32KB of read-only (flash) memory.

    Here's the layout of the graph:

    (fingerprint_input)
                    v
            [Conv2D]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
                [Relu]
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    This doesn't produce particularly accurate results, but it's designed to be
    used as the first stage of a pipeline, running on a low-energy piece of
    hardware that can always be on, and then wake higher-power chips when a
    possible utterance has been found, so that more accurate analysis can be done.

    During training, a dropout node is introduced after the relu, controlled by a
    placeholder.

    Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_4d = tf.reshape(fingerprint_input, [-1, width, height, 1])
    first_filter_width = 3
    first_filter_height = 3
    first_filter_count = 3 
    first_weights = tf.get_variable(
            name='first_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_filter_height, first_filter_width, 1, first_filter_count])
    first_bias = tf.get_variable(
            name='first_bias',
            initializer=tf.zeros_initializer,
            shape=[first_filter_count])
    first_conv_stride_x = 2
    first_conv_stride_y = 2
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                            [1, first_conv_stride_y, first_conv_stride_x, 1],
                            'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_dropout_shape = first_dropout.get_shape()
    first_dropout_output_width = first_dropout_shape[2]
    first_dropout_output_height = first_dropout_shape[1]
    first_dropout_element_count = int(
            first_dropout_output_width * first_dropout_output_height *
            first_filter_count)
    flattened_first_dropout = tf.reshape(first_dropout, [-1, first_dropout_element_count])
    label_count = output_shape
    final_fc_weights = tf.get_variable(
            name='final_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_dropout_element_count, label_count])
    final_fc_bias = tf.get_variable(
            name='final_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[label_count])
    final_fc = (tf.matmul(flattened_first_dropout, final_fc_weights) + final_fc_bias)
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_single_fc_model(fingerprint_input, input_size, output_size, is_training):
    """Builds a model with a single hidden fully-connected layer.

    This is a very simple model with just one matmul and bias layer. As you'd
    expect, it doesn't produce very accurate results, but it is very fast and
    simple, so it's useful for sanity testing.

    Here's the layout of the graph:

    (fingerprint_input)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = input_size
    label_count = output_size
    weights = tf.get_variable(
            name='weights',
            initializer=tf.truncated_normal_initializer(stddev=0.001),
            shape=[fingerprint_size, label_count])
    bias = tf.get_variable(
            name='bias', initializer=tf.zeros_initializer, shape=[label_count])
    logits = tf.matmul(fingerprint_input, weights) + bias
    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_two_fc_model(fingerprint_input, input_size, hidden_size, output_size, is_training):
    """Builds a model with two hidden fully-connected layer.

    Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = input_size
    label_count = output_size
    weights = tf.get_variable(
        name='weights',
        initializer=tf.truncated_normal_initializer(stddev=0.001),
        shape=[fingerprint_size, hidden_size])
    bias = tf.get_variable(
        name='bias', initializer=tf.zeros_initializer, shape=[hidden_size])
    weights2 = tf.get_variable(
        name='weights2',
        initializer=tf.truncated_normal_initializer(stddev=0.001),
        shape=[hidden_size, label_count])
    bias2 = tf.get_variable(
        name='bias2', initializer=tf.zeros_initializer, shape=[label_count])

    tmp = tf.nn.relu(tf.matmul(fingerprint_input, weights) + bias)
    logits = tf.nn.relu(tf.matmul(tmp, weights2) + bias2)

    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_three_fc_model(fingerprint_input, input_size, h1, h2, output_size, is_training):
    """Builds a model with two hidden fully-connected layer. """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    iw, ib = tf.truncated_normal_initializer(stddev=0.1), tf.zeros_initializer

    w1 = tf.get_variable('w1', [input_size, h1], initializer=iw)
    b1 = tf.get_variable('b1', [h1], initializer=ib)
    l1 = tf.nn.relu(tf.matmul(fingerprint_input, w1) + b1)
    
    w2 = tf.get_variable('w2', [h1, h2], initializer=iw)
    b2 = tf.get_variable('b2', [h2], initializer=ib)
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

    w3 = tf.get_variable('w3', [h2, output_size], initializer=iw)
    b3 = tf.get_variable('b3', [output_size], initializer=ib)
    logits = tf.matmul(l2, w3) + b3
    
    return logits, dropout_prob if is_training else logits


def create_conv_model(input_node, input_width, input_height, output_size, is_training):
    """Builds a standard convolutional model.

    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
                    v
            [Conv2D]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [Conv2D]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
                [Relu]
                    v
            [MaxPool]
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same
    paper with slightly less accuracy, see 'low_latency_conv' below.

    During training, dropout nodes are introduced after each relu, controlled by a
    placeholder.

    Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = input_width
    input_time_size = input_height
    fingerprint_4d = tf.reshape(input_node,
                                [-1, input_time_size, input_frequency_size, 1])
    # first_filter_width = 8
    # first_filter_height = 20
    first_filter_width = 3 
    first_filter_height = 3
    # first_filter_count = 64
    first_filter_count = 16 
    first_weights = tf.get_variable(
            name='first_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_filter_height, first_filter_width, 1, first_filter_count])
    first_bias = tf.get_variable(
            name='first_bias',
            initializer=tf.zeros_initializer,
            shape=[first_filter_count])
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                                                        'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 2 
    second_filter_height = 2
    second_filter_count = 16 
    second_weights = tf.get_variable(
            name='second_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[
                    second_filter_height, second_filter_width, first_filter_count,
                    second_filter_count
            ])
    second_bias = tf.get_variable(
            name='second_bias',
            initializer=tf.zeros_initializer,
            shape=[second_filter_count])
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                                                         'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
            second_conv_output_width * second_conv_output_height *
            second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout,
                                       [-1, second_conv_element_count])
    label_count = output_size 
    final_fc_weights = tf.get_variable(
            name='final_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[second_conv_element_count, label_count])
    final_fc_bias = tf.get_variable(
            name='final_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[label_count])
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                                                    is_training):
    """Builds a convolutional model with low compute requirements.

    This is roughly the network labeled as 'cnn-one-fstride4' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
                    v
            [Conv2D]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
                [Relu]
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    This produces slightly lower quality results than the 'conv' model, but needs
    fewer weight parameters and computations.

    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.

    Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                                            [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = input_time_size
    first_filter_count = 186
    first_filter_stride_x = 1
    first_filter_stride_y = 1
    first_weights = tf.get_variable(
            name='first_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_filter_height, first_filter_width, 1, first_filter_count])
    first_bias = tf.get_variable(
            name='first_bias',
            initializer=tf.zeros_initializer,
            shape=[first_filter_count])
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
            1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = math.floor(
            (input_frequency_size - first_filter_width + first_filter_stride_x) /
            first_filter_stride_x)
    first_conv_output_height = math.floor(
            (input_time_size - first_filter_height + first_filter_stride_y) /
            first_filter_stride_y)
    first_conv_element_count = int(
            first_conv_output_width * first_conv_output_height * first_filter_count)
    flattened_first_conv = tf.reshape(first_dropout,
                                                                        [-1, first_conv_element_count])
    first_fc_output_channels = 128
    first_fc_weights = tf.get_variable(
            name='first_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_conv_element_count, first_fc_output_channels])
    first_fc_bias = tf.get_variable(
            name='first_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[first_fc_output_channels])
    first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
    if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        second_fc_input = first_fc
    second_fc_output_channels = 128
    second_fc_weights = tf.get_variable(
            name='second_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_fc_output_channels, second_fc_output_channels])
    second_fc_bias = tf.get_variable(
            name='second_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[second_fc_output_channels])
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
    else:
        final_fc_input = second_fc
    label_count = model_settings['label_count']
    final_fc_weights = tf.get_variable(
            name='final_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[second_fc_output_channels, label_count])
    final_fc_bias = tf.get_variable(
            name='final_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[label_count])
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                                                    is_training, runtime_settings):
    """Builds an SVDF model with low compute requirements.

    This is based in the topology presented in the 'Compressing Deep Neural
    Networks using a Rank-Constrained Topology' paper:
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

    Here's the layout of the graph:

    (fingerprint_input)
                    v
                [SVDF]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
                [Relu]
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    This model produces lower recognition accuracy than the 'conv' model above,
    but requires fewer weight parameters and, significantly fewer computations.

    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.

    Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        The node is expected to produce a 2D Tensor of shape:
            [batch, model_settings['fingerprint_width'] *
                            model_settings['spectrogram_length']]
        with the features corresponding to the same time slot arranged contiguously,
        and the oldest slot at index [:, 0], and newest at [:, -1].
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.
        runtime_settings: Dictionary of information about the runtime.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.

    Raises:
            ValueError: If the inputs tensor is incorrectly shaped.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']

    # Validation.
    input_shape = fingerprint_input.get_shape()
    if len(input_shape) != 2:
        raise ValueError('Inputs to `SVDF` should have rank == 2.')
    if input_shape[-1].value is None:
        raise ValueError('The last dimension of the inputs to `SVDF` '
                                         'should be defined. Found `None`.')
    if input_shape[-1].value % input_frequency_size != 0:
        raise ValueError('Inputs feature dimension %d must be a multiple of '
                                         'frame size %d', fingerprint_input.shape[-1].value,
                                         input_frequency_size)

    # Set number of units (i.e. nodes) and rank.
    rank = 2
    num_units = 1280
    # Number of filters: pairs of feature and time filters.
    num_filters = rank * num_units
    # Create the runtime memory: [num_filters, batch, input_time_size]
    batch = 1
    memory = tf.get_variable(
            initializer=tf.zeros_initializer,
            shape=[num_filters, batch, input_time_size],
            trainable=False,
            name='runtime-memory')
    # Determine the number of new frames in the input, such that we only operate
    # on those. For training we do not use the memory, and thus use all frames
    # provided in the input.
    # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
    if is_training:
        num_new_frames = input_time_size
    else:
        window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                                                     model_settings['sample_rate'])
        num_new_frames = tf.cond(
                tf.equal(tf.count_nonzero(memory), 0),
                lambda: input_time_size,
                lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
    new_fingerprint_input = fingerprint_input[
            :, -num_new_frames*input_frequency_size:]
    # Expand to add input channels dimension.
    new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

    # Create the frequency filters.
    weights_frequency = tf.get_variable(
            name='weights_frequency',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[input_frequency_size, num_filters])
    # Expand to add input channels dimensions.
    # weights_frequency: [input_frequency_size, 1, num_filters]
    weights_frequency = tf.expand_dims(weights_frequency, 1)
    # Convolve the 1D feature filters sliding over the time dimension.
    # activations_time: [batch, num_new_frames, num_filters]
    activations_time = tf.nn.conv1d(
            new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
    # Rearrange such that we can perform the batched matmul.
    # activations_time: [num_filters, batch, num_new_frames]
    activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

    # Runtime memory optimization.
    if not is_training:
        # We need to drop the activations corresponding to the oldest frames, and
        # then add those corresponding to the new frames.
        new_memory = memory[:, :, num_new_frames:]
        new_memory = tf.concat([new_memory, activations_time], 2)
        tf.assign(memory, new_memory)
        activations_time = new_memory

    # Create the time filters.
    weights_time = tf.get_variable(
            name='weights_time',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[num_filters, input_time_size])
    # Apply the time filter on the outputs of the feature filters.
    # weights_time: [num_filters, input_time_size, 1]
    # outputs: [num_filters, batch, 1]
    weights_time = tf.expand_dims(weights_time, 2)
    outputs = tf.matmul(activations_time, weights_time)
    # Split num_units and rank into separate dimensions (the remaining
    # dimension is the input_shape[0] -i.e. batch size). This also squeezes
    # the last dimension, since it's not used.
    # [num_filters, batch, 1] => [num_units, rank, batch]
    outputs = tf.reshape(outputs, [num_units, rank, -1])
    # Sum the rank outputs per unit => [num_units, batch].
    units_output = tf.reduce_sum(outputs, axis=1)
    # Transpose to shape [batch, num_units]
    units_output = tf.transpose(units_output)

    # Appy bias.
    bias = tf.get_variable(
            name='bias', initializer=tf.zeros_initializer, shape=[num_units])
    first_bias = tf.nn.bias_add(units_output, bias)

    # Relu.
    first_relu = tf.nn.relu(first_bias)

    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    first_fc_output_channels = 256
    first_fc_weights = tf.get_variable(
            name='first_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[num_units, first_fc_output_channels])
    first_fc_bias = tf.get_variable(
            name='first_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[first_fc_output_channels])
    first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
    if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        second_fc_input = first_fc
    second_fc_output_channels = 256
    second_fc_weights = tf.get_variable(
            name='second_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_fc_output_channels, second_fc_output_channels])
    second_fc_bias = tf.get_variable(
            name='second_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[second_fc_output_channels])
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
    else:
        final_fc_input = second_fc
    label_count = model_settings['label_count']
    final_fc_weights = tf.get_variable(
            name='final_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[second_fc_output_channels, label_count])
    final_fc_bias = tf.get_variable(
            name='final_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[label_count])
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_tiny_conv_model(fingerprint_input, model_settings, is_training):
    """Builds a convolutional model aimed at microcontrollers.

    Devices like DSPs and microcontrollers can have very small amounts of
    memory and limited processing power. This model is designed to use less
    than 20KB of working RAM, and fit within 32KB of read-only (flash) memory.

    Here's the layout of the graph:

    (fingerprint_input)
                    v
            [Conv2D]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v
                [Relu]
                    v
            [MatMul]<-(weights)
                    v
            [BiasAdd]<-(bias)
                    v

    This doesn't produce particularly accurate results, but it's designed to be
    used as the first stage of a pipeline, running on a low-energy piece of
    hardware that can always be on, and then wake higher-power chips when a
    possible utterance has been found, so that more accurate analysis can be done.

    During training, a dropout node is introduced after the relu, controlled by a
    placeholder.

    Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

    Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['fingerprint_width']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 10
    first_filter_count = 8
    first_weights = tf.get_variable(
            name='first_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_filter_height, first_filter_width, 1, first_filter_count])
    first_bias = tf.get_variable(
            name='first_bias',
            initializer=tf.zeros_initializer,
            shape=[first_filter_count])
    first_conv_stride_x = 2
    first_conv_stride_y = 2
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                                                        [1, first_conv_stride_y, first_conv_stride_x, 1],
                                                        'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_dropout_shape = first_dropout.get_shape()
    first_dropout_output_width = first_dropout_shape[2]
    first_dropout_output_height = first_dropout_shape[1]
    first_dropout_element_count = int(
            first_dropout_output_width * first_dropout_output_height *
            first_filter_count)
    flattened_first_dropout = tf.reshape(first_dropout,
                                                                             [-1, first_dropout_element_count])
    label_count = model_settings['label_count']
    final_fc_weights = tf.get_variable(
            name='final_fc_weights',
            initializer=tf.truncated_normal_initializer(stddev=0.01),
            shape=[first_dropout_element_count, label_count])
    final_fc_bias = tf.get_variable(
            name='final_fc_bias',
            initializer=tf.zeros_initializer,
            shape=[label_count])
    final_fc = (
            tf.matmul(flattened_first_dropout, final_fc_weights) + final_fc_bias)
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
    Args:
        sess: TensorFlow session.
        start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)
