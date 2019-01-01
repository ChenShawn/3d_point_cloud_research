from baseline import Network as basenet
import tensorflow as tf
import numpy as np
import h5py, os

from datetime import datetime
from utils import conv1d, pooling1d, fully_connect, show_all_variables
from utils import Reader
from utils import FourierReader as fReader


class FourierNetwork(basenet):

    def __init__(self, input_shape, learning_rate=2e-4, batch_size=64,
                 model_dir='./checkpoint2/', log_dir='./logs/', pre_train=True):
        super(FourierNetwork, self).__init__(input_shape, learning_rate, batch_size,
                                             model_dir, log_dir, pre_train)
        self.name = 'Fourier_Network'

    def _build_model(self, input_op, is_training=True, keep_prob=0.5):
        with tf.variable_scope('baseline', reuse=not is_training):
            conv1_1 = tf.nn.relu(conv1d(input_op, n_out=64, name='conv1_1'))
            conv1_2 = tf.nn.relu(conv1d(conv1_1, n_out=64, name='conv1_2'))
            pool_1 = pooling1d(conv1_2, name='pool_1')

            conv2_1 = tf.nn.relu(conv1d(pool_1, n_out=128, name='conv2_1'))
            conv2_2 = tf.nn.relu(conv1d(conv2_1, n_out=128, name='conv2_2'))
            pool_2 = pooling1d(conv2_2, name='pool_2')

            conv3_1 = tf.nn.relu(conv1d(pool_2, n_out=256, name='conv3_1'))
            conv3_2 = tf.nn.relu(conv1d(conv3_1, n_out=256, name='conv3_2'))
            conv3_3 = tf.nn.relu(conv1d(conv3_2, n_out=256, name='conv3_3'))
            pool_3 = pooling1d(conv3_3, name='pool_3')

            conv4_1 = tf.nn.relu(conv1d(pool_3, n_out=512, name='conv4_1'))
            conv4_2 = tf.nn.relu(conv1d(conv4_1, n_out=512, name='conv4_2'))
            conv4_3 = tf.nn.relu(conv1d(conv4_2, n_out=512, name='conv4_3'))
            pool_4 = pooling1d(conv4_3, name='pool_4')

            conv5_1 = tf.nn.relu(conv1d(pool_4, n_out=512, name='conv5_1'))
            conv5_2 = tf.nn.relu(conv1d(conv5_1, n_out=256, name='conv5_2'))
            conv5_3 = tf.nn.relu(conv1d(conv5_2, n_out=256, name='conv5_3'))
            pool_5 = pooling1d(conv5_3, name='pool_5')

            shape = pool_5.get_shape()
            new_shape = [-1, shape[1].value * shape[2].value]
            reshaped = tf.reshape(pool_5, new_shape, name='reshaped')
            fc_1 = tf.nn.relu(fully_connect(reshaped, n_out=1024, name='fc_1'))
            fc1_dropped = tf.nn.dropout(fc_1, keep_prob=keep_prob, name='fc1_dropped')
            fc_2 = tf.nn.relu(fully_connect(fc1_dropped, n_out=256, name='fc_2'))
            fc2_dropped = tf.nn.dropout(fc_2, keep_prob=keep_prob, name='fc2_dropped')
            fc_3 = fully_connect(fc2_dropped, n_out=40, name='fc_3')
            return tf.nn.softmax(fc_3, name='softmax'), fc_3

    def train(self, num_iter, log_iter=40):
        reader = fReader()
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        for it in range(num_iter):
            batch_xs, batch_ys = reader.next_batch(self.batch_size)
            self.sess.run(self.optim, feed_dict={self.batch_xs: batch_xs, self.batch_ys: batch_ys})
            if it % log_iter == 20:
                loss, mAP, summaries, _ = self.sess.run(
                    [self.loss, self.mAP, self.summaries, self.update_acc],
                    feed_dict={self.batch_xs: batch_xs, self.batch_ys: batch_ys}
                )
                words = ' --Iteration %d --Loss: %g --Average precision: %g' % \
                        (self.counter, loss, mAP)
                print(str(datetime.now()) + words)
                writer.add_summary(summaries, self.counter)
            self.counter += 1
        print('Training finished!! Ready to save...')
        self.save()


if __name__ == '__main__':
    net = FourierNetwork([2048, 3], batch_size=128, pre_train=True)
    # net.train(10000, log_iter=100)
    # print('Done!')

    reader = fReader(is_test=False)
    print('The model accuracy is {}'.format(net.test(reader)))
