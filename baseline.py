import tensorflow as tf
import numpy as np
import h5py, os

from datetime import datetime
from utils import conv1d, pooling1d, fully_connect, show_all_variables
from utils import Reader as uReader

class Network(object):
    name = 'baseline_network'

    def __init__(self, input_shape, learning_rate=2e-4, batch_size=64,
                 model_dir='./checkpoint/', log_dir='./logs/', pre_train=True):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.model_dir = model_dir

        self.batch_xs = tf.placeholder(tf.float32, [self.batch_size] + input_shape, name='batch_xs')
        self.batch_ys = tf.placeholder(tf.uint8, [self.batch_size], name='batch_ys')
        self.labels = tf.cast(self.batch_ys, dtype=tf.int32, name='labels')

        _, self.logits = self._build_model(self.batch_xs, is_training=True, keep_prob=0.5)
        y_pred, _ = self._build_model(self.batch_xs, is_training=False, keep_prob=1.0)
        self.y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.mAP = self._build_map(self.labels, self.y_pred)
        self.accuracy, self.update_acc = tf.metrics.accuracy(self.labels, self.y_pred)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                              labels=self.labels)
        self.loss = tf.reduce_mean(loss)
        self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        loss_summary = tf.summary.scalar('loss', self.loss)
        ap_summary = tf.summary.scalar('average precision', self.accuracy)
        self.summaries = tf.summary.merge([loss_summary, ap_summary], name='summaries')
        show_all_variables()

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

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
            pool_3 = pooling1d(conv3_2, name='pool_3')

            conv4_1 = tf.nn.relu(conv1d(pool_3, n_out=256, name='conv4_1'))
            conv4_2 = tf.nn.relu(conv1d(conv4_1, n_out=256, name='conv4_2'))
            pool_4 = pooling1d(conv4_2, name='pool_4')
            
            conv5_1 = tf.nn.relu(conv1d(pool_4, n_out=256, name='conv5_1'))
            conv5_2 = tf.nn.relu(conv1d(conv5_1, n_out=256, name='conv5_2'))
            pool_5 = pooling1d(conv5_2, name='pool_5')

            shape = pool_5.get_shape()
            new_shape = [-1, shape[1].value * shape[2].value]
            reshaped = tf.reshape(pool_5, new_shape, name='reshaped')
            fc_1 = tf.nn.relu(fully_connect(reshaped, n_out=1024, name='fc_1'))
            fc1_dropped = tf.nn.dropout(fc_1, keep_prob=keep_prob, name='fc1_dropped')
            fc_2 = tf.nn.relu(fully_connect(fc1_dropped, n_out=256, name='fc_2'))
            fc2_dropped = tf.nn.dropout(fc_2, keep_prob=keep_prob, name='fc2_dropped')
            fc_3 = fully_connect(fc2_dropped, n_out=40, name='fc_3')
            return tf.nn.softmax(fc_3, name='softmax'), fc_3

    @staticmethod
    def _build_map(label, pred):
        correct = tf.cast(tf.equal(label, pred), dtype=tf.float32)
        return tf.reduce_mean(correct)

    def train(self, num_iter, log_iter=40):
        reader = uReader(is_test=False)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        for it in range(num_iter):
            batch_xs, batch_ys = reader.next_batch(self.batch_size, ordered=False)
            self.sess.run(self.optim, feed_dict={self.batch_xs: batch_xs, self.batch_ys: batch_ys})
            if it % log_iter == 20:
                loss, mAP, summaries = self.sess.run(
                    [self.loss, self.mAP, self.summaries],
                    feed_dict={self.batch_xs: batch_xs, self.batch_ys: batch_ys}
                )
                words = ' --Iteration %d --Loss: %g --Average precision: %g' % \
                        (self.counter, loss, mAP)
                print(str(datetime.now()) + words)
                writer.add_summary(summaries, self.counter)
            self.counter += 1
        print('Training finished!! Ready to save...')
        self.save()

    def test(self, reader=None):
        if reader is None:
            reader = uReader(is_test=True)
        acc = 0.0
        while True:
            try:
                batch_xs, batch_ys = reader.next_batch(self.batch_size, ordered=True)
                acc, _ = self.sess.run([self.accuracy, self.update_acc],
                                       feed_dict={self.batch_xs:batch_xs, self.batch_ys: batch_ys})
            except IndexError as err:
                print(err.args)
                break
        return acc

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif len(os.listdir(self.model_dir)) != 0:
            fs = os.listdir(self.model_dir)
            for f in fs:
                os.remove(self.model_dir + f)
        save_path = self.saver.save(self.sess, self.model_dir + self.name, global_step=self.counter)
        print('MODEL RESTORED IN: ' + save_path)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, self.model_dir + ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    import sys
    train_iter = int(sys.argv[1])
    net = Network([2048, 3], batch_size=128, pre_train=True)
    if train_iter != 0:
        net.train(train_iter, log_iter=50)
        print('Done!!')

    print('The accuracy of the model is {}'.format(net.test()))
    print('Done!!')
