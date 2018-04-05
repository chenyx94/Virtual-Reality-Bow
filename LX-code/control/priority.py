import tensorflow as tf
import numpy as np
class Priority(object):
    def __init__(self, model):
        self.model = model
        return
    def fetch(self):
        return self.model.calc_priority()

class PriorityModel(object):
    def __init__(self, s_dim, a_dim, H_units):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.H_units = H_units
        self.s = tf.placeholder(tf.float32, [None, self.s_dim])
        self.discounted_rewards = tf.placeholder(tf.float32, [None, ])
        self.a = tf.placeholder(tf.int32, [None, ])


    def calc_priority(self):
        pi = self.sess.run(self.logp, feed_dict=[])
    def build_model(self):
        with tf.name_scope('REINFORCE'):
            with tf.name_scope('hidden_1'):
                w1 = tf.Variable(tf.div(tf.random_normal([self.s_dim, self.H_units]),np.sqrt(self.s_dim)))
                b1 = tf.Variable(tf.constant(0.0, shape=[self.H_units]))
                h1_raw = tf.nn.relu(tf.matmul(self.s_dim, w1) + b1)
                mean_pool = tf.reduce_mean(h1_raw, axis=0)
                max_pool = tf.reduce_max(h1_raw, axis=0)
                h1 = tf.concat([mean_pool, max_pool], axis=0)
            with tf.name_scope('hidden_2'):
                w2 = tf.Variable(tf.div(tf.random_normal([2 * self.H_units, self.action_dim]), np.sqrt(self.H_units)))
                b2 = tf.Variable(tf.constant(0.0, shape=[ self.action_dim]))
                self.logp = tf.matmul(h1, w2) + b2
            # optimizer
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=1e-4, decay=0.99)
            # loss
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logp, labels=self.a))

    def learn(self):

            # gradient
            self.gradient = self.optimizer.compute_gradients(self.loss)
            # policy gradient
            for i, (grad, var) in enumerate(self.gradient):
                if grad is not None:
                    pg_grad = grad * self.discounted_rewards
                    # gradient clipping
                    pg_grad = tf.clip_by_value(
                        pg_grad, -self.max_gradient, self.max_gradient)
                    self.gradient[i] = (pg_grad, var)
            # train operation (apply gradient)
            self.train_op = self.optimizer.apply_gradients(self.gradient)