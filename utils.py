from tensorflow.contrib import slim
import numpy as np
import ctypes as ct
import tensorflow as tf
import math, cv2, os, h5py
from functools import reduce
from datetime import datetime

# arr should be 4 dimensional
def save_image(arr, name, idx, scale=True, path='./mnist/generated/'):
    if scale:
        arr = arr * (255.0 / np.max(arr))
    for i in range(arr.shape[0]):
        img_to_save = arr[i, :, :, :].astype(np.uint8)
        cv2.imwrite(path + str(idx) + '_' + str(i) + '_' + name, img_to_save)
    print('SAVING GENERATED IMAGES TO: ' + path + name)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def batch_norm(input_op, name, is_training, epsilon=1e-5, momentum=0.99):
    return tf.contrib.layers.batch_norm(input_op, decay=momentum, updates_collections=None,
                                        epsilon=epsilon, scale=True, is_training=is_training, scope=name)

def show_all_variables():
    all_variables = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)

def lrelu(input_op, leak=0.2, name='linear'):
    return tf.maximum(input_op, leak*input_op, name=name)

def conv2d(input_op, n_out, name, kh=5, kw=5, dh=2, dw=2, bias=True):
    n_in = input_op.get_shape()[-1].value
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_2d_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        if bias:
            biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, biases)
        else:
            return conv

def conv1d(input_op, n_out, name, ksize=5, dsize=1, bias=True):
    '''
    :param input_op: the input size should be [batch_size, sequence_length, channel_size].
    :param bias: the parameter should be specified as False when
     using batch_norm before ReLU, setting it to True will not make
     the program fail but waste the memory.
    '''
    n_in = input_op.get_shape()[-1].value
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_1d_w',
                                 shape=(ksize, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv1d(input_op, kernel, dsize, padding='SAME')
        if bias:
            biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, biases)
        else:
            return conv

def deconv2d(input_op, output_shape, kh=5, kw=5, dh=2, dw=2, name='deconv', bias_init=0.0):
    n_in = input_op.get_shape()[-1].value
    n_out = output_shape[-1]
    # filter : [height, width, output_channels, in_channels]
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernels',
                                 shape=(kh, kw, n_out, n_in),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                        output_shape=output_shape,
                                        strides=(1, dh, dw, 1))
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.nn.bias_add(deconv, biases)

def pooling(input_op, name, kh=2, kw=2, dh=2, dw=2, pooling_type='max'):
    if 'max' in pooling_type:
        return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)
    else:
        return tf.nn.avg_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def pooling1d(input_op, name, ksize=2, dsize=2):
    return tf.layers.max_pooling1d(input_op, pool_size=ksize, strides=dsize, name=name)

def fully_connect(input_op, n_out, name='fully_connected', bias_init=0.0):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='weights',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        biases = tf.get_variable(name='bias', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.matmul(input_op, kernel) + biases

def random_mask(input_op, batch_size=64):
    masks = [tf.ones([1, 32, 32, 3], dtype=tf.float32) for _ in range(batch_size)]
    off_h = tf.random_uniform([batch_size], 0, 32, dtype=tf.int32)
    off_w = tf.random_uniform([batch_size], 0, 32, dtype=tf.int32)
    paddings = [tf.image.pad_to_bounding_box(masks[it], off_h[it], off_w[it], 96, 96)
                for it in range(batch_size)]
    masking = 1.0 - tf.cast(tf.concat(paddings, axis=0), dtype=tf.float32)
    return input_op * masking

def gen_graphs(data, k=20):
    '''
    :param data: float32 np.ndarray which has shape (batch_size, num_points, 3)
    :param k: an integer representing the number of the nearest neighbor for constructing the graph
    :return: numbers of float32 np.ndarray type adjacency matrices of the graph
    '''
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'graph_op.so'), '.')
    num_graphs = data.shape[0]
    num_point = data.shape[1]
    c_num_graphs = ct.c_int(num_graphs)
    c_num_point = ct.c_int(num_point)
    c_data = data.ctypes.data_as(ct.c_void_p)
    adjacency = np.zeros((num_graphs, num_point, num_point), dtype=np.float32)
    c_adjacency = adjacency.ctypes.data_as(ct.c_void_p)
    c_k = ct.c_int(k)
    dll.gen_graphs(c_data, c_num_point, c_num_graphs, c_k, c_adjacency)
    return adjacency


class Reader(object):
    default_base_dir = '/home/zcx/Documents/datasets/ModelNet/'

    def __init__(self, is_test=False):
        train_name = ['ply_data_train{}.h5'.format(it) for it in range(5)]
        test_name = ['ply_data_test0.h5', 'ply_data_test1.h5']
        self.train_name = [os.path.join(self.default_base_dir, name) for name in train_name]
        self.label_name = open('/home/zcx/Documents/datasets/ModelNet/shape_names.txt', 'r').read().splitlines()
        self.test_name = [os.path.join(self.default_base_dir, name) for name in test_name]
        if not is_test:
            self.data = [h5py.File(name, 'r') for name in self.train_name]
        else:
            self.data = [h5py.File(name, 'r') for name in self.test_name]
        self.counter = 0; self.file_counter = 0
        print(self)

    def next_batch(self, num, ordered=False):
        '''
        :param num: The number of data to be sampled.
        :param ordered: set this parameter to True when testing
        :return: [num, num_points, 3], [num]
        '''
        if not ordered:
            file_idx = np.random.randint(0, len(self.data))
            data_idx = np.random.randint(0, self.data[file_idx]['data'].shape[0], [num])
            batch_xs = self.data[file_idx]['data'].value[data_idx, :, :]
            batch_ys = self.data[file_idx]['label'].value[data_idx]
            return batch_xs, np.squeeze(batch_ys)
        else:
            end_idx = self.counter + num
            batch_xs = self.data[self.file_counter]['data'].value[self.counter: end_idx, :, :]
            batch_ys = self.data[self.file_counter]['label'].value[self.counter: end_idx]
            self.counter = end_idx
            if end_idx >= self.data[self.file_counter]['data'].shape[0]:
                self.file_counter = (self.file_counter + 1) % 2
                if self.file_counter == 0:
                    raise IndexError('Testing data delivery finished!!')
                self.counter = num - batch_xs.shape[0]
                app_data = self.data[self.file_counter]['data'].value[0: self.counter, :, :]
                app_label = self.data[self.file_counter]['label'].value[0: self.counter]
                batch_xs = np.concatenate([batch_xs, app_data], axis=0)
                batch_ys = np.concatenate([batch_ys, app_label], axis=0)
            return batch_xs, np.squeeze(batch_ys)

    def Data(self, idx=0):
        return self.data[idx]['data'].value, self.data[idx]['label'].value

    def __str__(self):
        ubound = '----------------POINT CLOUD READER------------------'
        lbound = '----------------------------------------------------'
        info_1 = 'Reader initialized in directory: {}'.format(self.default_base_dir)
        info_2 = 'Files loaded: {}'.format(len(self.data))
        info_3 = 'Data loaded: {}'.format(
            reduce(lambda x, y: x + y, [item['data'].shape[0] for item in self.data])
        )
        info_4 = 'Data shape: ' + str(self.data[0]['data'].shape[1 :])
        return '\n'.join([ubound, info_1, info_2, info_3, info_4, lbound])

    def __del__(self):
        for item in self.data:
            item.close()


class FourierReader(object):
    default_base_dir = '/home/zcx/Documents/datasets/ModelNet/'

    def __init__(self, is_test=False):
        train_name = ['ply_data_train{}.h5'.format(it) for it in range(5)]
        test_name = ['ply_data_test0.h5', 'ply_data_test1.h5']
        self.train_name = [os.path.join(self.default_base_dir, name) for name in train_name]
        self.label_name = open('/home/zcx/Documents/datasets/ModelNet/shape_names.txt', 'r').read().splitlines()
        self.test_name = [os.path.join(self.default_base_dir, name) for name in test_name]
        if not is_test:
            self.data = [h5py.File(name, 'r') for name in self.train_name]
        else:
            self.data = [h5py.File(name, 'r') for name in self.test_name]
        self.counter = 0; self.file_counter = 0
        if not is_test:
            self.arrays = [np.load('./fourier_data{}.npy'.format(it)) for it in range(5)]
        else:
            self.arrays = [np.load('./fourier_test{}.npy'.format(it)) for it in range(2)]
        print(self)

    def __str__(self):
        ubound = '----------------POINT CLOUD READER------------------'
        lbound = '----------------------------------------------------'
        info_1 = 'Reader initialized in directory: {}'.format(self.default_base_dir)
        info_2 = 'Files loaded: {}'.format(len(self.arrays))
        info_3 = 'Data loaded: {}'.format(
            reduce(lambda x, y: x + y, [item.shape[0] for item in self.arrays])
        )
        info_4 = 'Data shape: ' + str(self.arrays[0].shape[1 :])
        return '\n'.join([ubound, info_1, info_2, info_3, info_4, lbound])

    def next_batch(self, num, ordered=False):
        if not ordered:
            file_idx = np.random.randint(0, len(self.arrays))
            data_idx = np.random.randint(0, self.arrays[file_idx].shape[0], [num])
            batch_xs = self.arrays[file_idx][data_idx, :, :]
            batch_ys = self.data[file_idx]['label'].value[data_idx]
            return batch_xs, np.squeeze(batch_ys)
        else:
            end_idx = self.counter + num
            batch_xs = self.arrays[self.file_counter][self.counter: end_idx, :, :]
            batch_ys = self.data[self.file_counter]['label'].value[self.counter: end_idx]
            self.counter = end_idx
            if end_idx >= self.arrays[self.file_counter].shape[0]:
                self.file_counter = (self.file_counter + 1) % 2
                if self.file_counter == 0:
                    raise IndexError('Testing data delivery finished!!')
                self.counter = num - batch_xs.shape[0]
                app_data = self.arrays[self.file_counter][0: self.counter, :, :]
                app_label = self.data[self.file_counter]['label'].value[0: self.counter]
                batch_xs = np.concatenate([batch_xs, app_data], axis=0)
                batch_ys = np.concatenate([batch_ys, app_label], axis=0)
            return batch_xs, np.squeeze(batch_ys)


if __name__ == '__main__':
    reader = FourierReader()
    begin = datetime.now()
    for it in range(10):
        xs, ys = reader.next_batch(64)
        print(xs.shape, ys.shape, end=' || ')
    end = datetime.now()
    print(end - begin)
