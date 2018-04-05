# -*-coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from .Config import *
from .memory import *
from tensorflow.contrib import rnn
from Memory import *
import math
import copy

CONFIG = Config()

LR_A = CONFIG.lr_a  # learning rate for actor
LR_C = CONFIG.lr_c  # learning rate for critic
GAMMA = CONFIG.gamma  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = CONFIG.memory_capacity
BATCH_SIZE = CONFIG.batch_size
REPLAY_START_SIZE=CONFIG.replay_start_size


class DDPG(object):
    def __init__(self, sess, a_dim, s_dim, a_bound, actor, critic):
        self.memory = np.zeros((MEMORY_CAPACITY), dtype=object)
        self.pointer = 0
        self.sess = sess
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S1 = tf.placeholder(tf.float32, [None, s_dim], 's1')
        self.global_step = tf.Variable(0, trainable=False)
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.S1_ = tf.placeholder(tf.float32, [None, s_dim], 'S1_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.unit_size = tf.placeholder(tf.int32, name='unit_size')
        self.unit_size_ = tf.placeholder(tf.int32, name='unit_size_')
        self.variable_summaries('reward', self.R)
        with tf.variable_scope(actor):
            self.a = self._build_a(self.S, self.S1, scope='eval', trainable=True, unit_size=self.unit_size)
            self.variable_summaries('eval_a', self.a)
            a_ = self._build_a(self.S_, self.S1_, scope='target', trainable=False, unit_size=self.unit_size_)
            self.variable_summaries('target_a', a_)
        with tf.variable_scope(critic):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True, unit_size=self.unit_size)
            self.variable_summaries('eval_q', q)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False, unit_size=self.unit_size_)
            self.variable_summaries('target_q', q_)
        # networks parameters


        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor + '/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor + '/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=critic + '/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=critic + '/target')

        for var in self.ae_params:
            self.variable_summaries(var.op.name, var)
        for var in self.at_params:
            self.variable_summaries(var.op.name, var)
        for var in self.ce_params:
            self.variable_summaries(var.op.name, var)
        for var in self.ct_params:
            self.variable_summaries(var.op.name, var)

        q_target = self.R + GAMMA * q_
        self.variable_summaries('q_target', q_target)
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params,
                                                            global_step=self.global_step)
        a_loss = - tf.reduce_sum(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params,
                                                            global_step=self.global_step)
        tf.summary.scalar('a_loss', a_loss)
        tf.summary.scalar('c_loss', td_error)

        self.merged_summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=CONFIG.max_to_keep)
#        self.summary_writer = tf.summary.FileWriter(CONFIG.summary_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def variable_summaries(self, name, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '_mean', mean)
            with tf.name_scope(name + '_stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar(name + '_stddev', stddev)
            tf.summary.scalar(name + '_max', tf.reduce_max(var))
            tf.summary.scalar(name + '_min', tf.reduce_min(var))
            tf.summary.histogram(name + '_histogram', var)

    def choose_action(self, s, s1, unit_size):
        a_out = self.sess.run(self.a, {self.S: s,self.S1:s1, self.unit_size: unit_size})

        return a_out

    def learn(self):
        # hard replace parameters
        if self.a_replace_counter % REPLACE_ITER_A == 0:#actor target networks更新
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        if self.c_replace_counter % REPLACE_ITER_C == 0:#critic target network更新
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
        self.a_replace_counter += 1
        self.c_replace_counter += 1
        if self.pointer >= MEMORY_CAPACITY:
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        elif self.pointer >= REPLAY_START_SIZE:
            indices = np.random.choice(self.pointer, size=BATCH_SIZE)#改为从start开始学
        else:
            print("replay_start_size is smaller than batch_size!")
            return
        bt = self.memory[indices]
        for i in range(bt.size): #原来是这个for循环大大增加了运行时间。
            self.sess.run([self.atrain], {self.S:bt[i].s, self.S1:bt[i].s1,
                                        self.unit_size:bt[i].unit_size})
            reward = np.array(bt[i].r).reshape((-1, 1))#这里张煜reshape了一下。其实不仅reward要reshape，state最好也要reshape
            summary, _ = self.sess.run([self.merged_summary_op, self.ctrain], {self.S: bt[i].s, self.S1:bt[i].s1,
                                                                               self.a: bt[i].a,
                                                                               self.R: reward, self.S_: bt[i].s_,
                                                                               self.S1_:bt[i].s1_,
                                                                               self.unit_size: bt[i].unit_size,
                                                                               self.unit_size_:bt[i].unit_size_})
#            self.summary_writer.add_summary(summary, self.sess.run(self.global_step))

    def store_transition(self, s, s1, a, r, s_, s1_, unit_size, unit_size_):
        mymemory = Memory(s, s1, a, r, s_, s1_, unit_size, unit_size_)
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index] = mymemory
        self.pointer += 1

    def _build_a(self, s, s1, scope, trainable, unit_size):
        with tf.variable_scope(scope):  # s:[enemy_size+unit_size,hidden_size]
            w1 = tf.Variable(
                tf.truncated_normal([self.s_dim, CONFIG.hidden_size * 2], stddev=CONFIG.std, dtype=tf.float32),
                trainable=trainable)  # 2*CONFIG.hidden_size
            b1 = tf.Variable(tf.zeros([CONFIG.hidden_size * 2, ]))
            x = tf.nn.elu(tf.matmul(s, w1) + b1)  # [unit_size+enemy_size,2*hidden_size]
            x1 = tf.nn.elu(tf.matmul(s1, w1) + b1)
            mean_pool = tf.reduce_mean(x, axis=0)
            max_pool = tf.reduce_max(x, axis=0)
            feature = tf.concat([mean_pool, max_pool], axis=0)
            feature = tf.reshape(feature, [1, 4 * CONFIG.hidden_size])
            feature = tf.multiply(feature, tf.zeros([unit_size, 1]) + 1)  # unit_size,4*config.hidden_size
            feature = tf.concat([x1, feature], axis=1)
            final_feature = tf.reshape(feature, [1, unit_size, 6 * CONFIG.hidden_size])

            cell = tf.contrib.rnn.BasicLSTMCell(num_units=CONFIG.hidden_size)
#            cell = tf.contrib.rnn.BasicRNNCell(num_units=CONFIG.hidden_size)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=final_feature,
                                                              dtype=tf.float32, scope=scope)
            outputs_all = tf.concat(outputs, 1)
            outputs_all = tf.reshape(outputs_all, [unit_size, 2 * CONFIG.hidden_size])
            w2 = tf.Variable(
                tf.truncated_normal([2 * CONFIG.hidden_size, self.a_dim], stddev=CONFIG.std, dtype=tf.float32),
                trainable=trainable)
            b2 = tf.Variable(tf.zeros([self.a_dim, ], dtype=tf.float32), trainable=trainable)
            action_value = tf.nn.tanh(tf.matmul(outputs_all, w2) + b2)
            return action_value

    def _build_c(self, s, a, scope, trainable, unit_size):
        with tf.variable_scope(scope):
            w1 = tf.Variable(
                tf.truncated_normal([self.s_dim, CONFIG.hidden_size * 2], stddev=CONFIG.std, dtype=tf.float32),
                trainable=trainable)  # 10*CONFIG.hidden_size
            b1 = tf.Variable(tf.zeros([CONFIG.hidden_size * 2, ]))
            x = tf.nn.elu(tf.matmul(s, w1) + b1)  # [unit_size+enemy_size,10*hidden_size]
            mean_pool = tf.reduce_mean(x, axis=0)
            max_pool = tf.reduce_max(x, axis=0)
            feature = tf.concat([mean_pool, max_pool], axis=0)
            feature = tf.reshape(feature, [1, 4 * CONFIG.hidden_size])
            feature = tf.multiply(feature, tf.zeros([unit_size, 1]) + 1)  # unit_size,4*config.hidden_size
            action_and_q = tf.concat([a, feature], axis=1)  # unit_size,4*config.hidden_size+3
            input_feature = tf.reshape(action_and_q, [1, unit_size, CONFIG.hidden_size * 4 + 3])
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=CONFIG.hidden_size)
#            cell = tf.contrib.rnn.BasicRNNCell(num_units=CONFIG.hidden_size)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=input_feature,
                                                              dtype=tf.float32, scope=scope)
            outputs = tf.concat(outputs, 1)
            outputs = tf.reshape(outputs, [unit_size, 2 * CONFIG.hidden_size])
            w2 = tf.Variable(tf.truncated_normal([2 * CONFIG.hidden_size, 1], stddev=CONFIG.std, dtype=tf.float32),
                             trainable=trainable)
            b2 = tf.Variable(tf.zeros([1, ], dtype=tf.float32), trainable=trainable)
            Q_value = tf.matmul(outputs, w2) + b2
            return Q_value  # Q(s,a)

def softmax(y):
    """ simple helper function here that takes unnormalized logprobs """
    maxy = np.amax(y)
    e = np.exp(y - maxy)
    return e / np.sum(e)

class DQN(object):
    """docstring for DQN"""
    def __init__(self, session, resolution=(10,14), command_size = 14, learning_rate=5e-3, buffer_size=10000, replace_target_iter=100,gamma=0.99,batch_size=64, index=0, deeper=False):
        super(DQN, self).__init__()
        print("DQN init begin")
        self.hidden1_units = 100
        self.hidden2_units = 100
        self.hidden3_units = 100
        self.hidden4_units = 100
        self.lr = learning_rate
        self.sess = session
        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0
        self.gamma = gamma
        self.index = index
        self.resolution = resolution
        self.command_size = command_size
        print("before build net")
        self._build_net()
        print("DQN not normal!!!!!!!!!!!!!!!!!!!!!")
        print([v.name for v in self.t_params]) #=> prints lists of vars created
        print([v.name for v in self.e_params]) #=> prints lists of vars created
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        self.saver = tf.train.Saver(max_to_keep=CONFIG.max_to_keep)
        self.sess.run(tf.global_variables_initializer())
        
        

        
    def _build_net(self):
        def fully_connected(prev_layer, num_units, is_training):
            layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
            layer = tf.nn.relu(layer)
            return layer
        def conv2d(prev_layer, filters, kernel_size,strides, pool_size,pool_stride,is_training):
            conv = tf.layers.conv2d(   
                inputs=prev_layer,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=tf.nn.relu
            )           
            # pool = tf.layers.max_pooling2d(
            #     conv,
            #     pool_size=pool_size,
            #     strides=pool_stride,
            # )    
            return conv
        def build_layers(scope, s):
            print("begin build_layers scope:{} .....".format(scope))
            with tf.variable_scope(scope+str(self.index)):
                # conv0 = conv2d(s,filters=32,kernel_size=16,strides=4,pool_size=2,pool_stride=2,is_training=self.is_training)
                # shape (64, 64, 1)
                print("after build conv0")
                conv1 = conv2d(s,filters=32,kernel_size=8,strides=4,pool_size=2,pool_stride=2,is_training=self.is_training)
                # conv1 (18,18,16)
                print("after build conv1")
                conv2 = conv2d(conv1,filters=64,kernel_size=4,strides=2,pool_size=2,pool_stride=2,is_training=self.is_training)
                # conv2 (13,13,32)
                print("after build conv2")
                conv3 = conv2d(conv2,filters=64,kernel_size=3,strides=1,pool_size=2,pool_stride=2,is_training=self.is_training)
                print("after build conv3")
                flatten_view = tf.contrib.layers.flatten(conv3)
                fc1 = fully_connected(flatten_view, num_units=128, is_training=self.is_training)
                fc2 = fully_connected(fc1, num_units=128, is_training=self.is_training)
                fc3 = fully_connected(fc2, num_units=64, is_training=self.is_training)

                q = tf.layers.dense(fc3, self.command_size, use_bias=True, activation=None)
            print("after build_layers scope:{}!".format(scope))
            return q
        print("begin init placeholders..")
        with tf.variable_scope('variable'+str(self.index)):
            self.is_training = tf.placeholder(tf.bool)
            self.a = tf.placeholder(tf.int64,[None], name="a")
            
            # ------------------ build evaluate_net ------------------
            self.s = tf.placeholder(tf.float32, [None] + list(self.resolution) + [1], name="State")  # input
            self.q_target = tf.placeholder(tf.float32, [None,self.command_size], name="TargetQ")  # for calculating loss
            # ------------------ build target_net ------------------
            self.s_ = tf.placeholder(tf.float32, [None] + list(self.resolution) + [1], name="State_next")  # input
        print("after init placeholders!")

        self.q_eval = build_layers("eval_net",self.s)
        self.q_next = build_layers("target_net",self.s_)
        print("before get params")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net'+str(self.index))
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net'+str(self.index))
        print("after get params")
        with tf.variable_scope('loss'+str(self.index)):

            self.prob_eval = tf.nn.softmax(self.q_eval)
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.prob_eval, tf.one_hot(self.a,self.command_size)))


        with tf.variable_scope('train'+str(self.index)):
            self.learning_rate = tf.placeholder(tf.float32)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            self.gradients = tf.gradients(self.loss,self.e_params)
            self._train_op = optimizer.apply_gradients(zip(self.gradients, self.e_params))


    def set_e_params(self,t_params):
        self.target_replace_op_e = [tf.assign(e, t) for e, t in zip(self.e_params, t_params)]
        self.sess.run(self.target_replace_op_e)
    def set_t_params(self,t_params):
        self.target_replace_op_t = [tf.assign(e, t) for e, t in zip(self.t_params, t_params)]
        self.sess.run(self.target_replace_op_t)

    def get_e_params(self):
        return self.sess.run(self.e_params)
    def get_t_params(self):
        return self.sess.run(self.t_params)

    def set_lr(self,learning_rate,lr_decay_rate,min_lr,decay_round):
        self.lr = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.min_lr = min_lr
        self.decay_round = decay_round
    def set_decay_params(self,DECAY_PARAMS):
        self.True_DECAY_PARAMS = DECAY_PARAMS
    def validation(self,s,a, dead_e_index):
        #----------------------mean zero begin---------------------------------
        # print("s.shape:{}".format(np.array(s).shape))
        s=np.array(s).reshape([-1,self.resolution[0],self.resolution[1],1])
        #----------------------mean zero end------------------------------------------
        q_eval = self.sess.run([self.q_eval], feed_dict={self.s:s,
                                                                self.is_training:False})

        q_eval = q_eval[0]
        # print("q_eval.shape:{}".format(np.array(q_eval).shape))
        # print("dead_e_index.shape:{}".format(np.array(dead_e_index).shape))
        # for i in range(0,30):
        #     print("dead_index[{}]:{}".format(i,dead_e_index[i]))
        acc_cnt = 0
        for i in range(0,len(s)):
            best_a = np.argmax(q_eval[i])
            if dead_e_index != None:
                while True:
                    if best_a <= 8:
                        break
                    index = best_a - 9
                    if index in dead_e_index[i]:
                        
                        q_eval[i][best_a] = -200000
                        best_a = np.argmax(q_eval[i])
                        index = best_a - 9
                        # print("index:",index)
                    else:
                        break
            if best_a == a[i]:
                acc_cnt+=1
        acc_rate = acc_cnt*1.0/len(s)*100
        return acc_rate

    def learn_with_one_episode(self,s,a,s_,isterminal,r,command_size, dead_e_index=None,double_dqn=True, t_prob_eval=None,use_old=None):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
        # print("len(s):{}".format(len(s)))
        # if self.learn_step_counter % self.decay_round == 0 and self.lr > self.min_lr:
        #     self.lr *= self.lr_decay_rate
        # if use_old==False:
        #     self.DECAY_PARAMS = 0
        # else:
        #     self.DECAY_PARAMS = self.True_DECAY_PARAMS
        loss_sum = 0
        lr = 0
        target_q = []
        q_eval_list = []
        q_next_list = []
        q_target = self.sess.run(self.q_eval,feed_dict={self.s:s,self.is_training:False})
        if double_dqn:
            q_eval_list = self.sess.run(self.q_eval,feed_dict={self.s:s_,self.is_training:False})

            q_next_list = self.sess.run(self.q_next,feed_dict={self.s_:s_,self.is_training:False})
        else:
            q_next_list = self.sess.run(self.q_next,feed_dict={self.s_:s_,self.is_training:False})

        for i in range(0,len(s)):
            if not bool(isterminal[i]):
                if double_dqn:
                    q_next = q_next_list[i]
                    q_eval = q_eval_list[i]

                    q2 = q_next[ np.argmax(q_eval) ]
                else:

                    q_next = q_next_list[i]
                    q2 = np.max(q_next)
                tmp_target_q = r[i]  +  q2
            else:
                tmp_target_q = r[i] 
            target_q.append(tmp_target_q)

        batch_size = len(s)
        batch_index = np.arange(batch_size,dtype=np.int32)
        q_target[batch_index,a] = target_q 

        cost,_,q_eval = self.sess.run([self.loss,self._train_op, self.q_eval], 
                                        feed_dict={ self.s:s,
                                                    self.a:a,
                                                    self.q_target:q_target, 
                                                    self.is_training:True,
                                                    self.learning_rate:self.lr})
        acc_rate = 0

        acc_cnt = 0
        for i in range(0,len(s)):
            best_a = np.argmax(q_eval[i])
            if dead_e_index != None:
                while True:
                    if best_a <= 8:
                        break
                    index = best_a - 9
                    if index in dead_e_index[i]:
                        q_eval[i][best_a] = -200000
                        best_a = np.argmax(q_eval[i])
                        index = best_a - 9
                    else:
                        break
            if best_a == a[i]:
                acc_cnt+=1
        acc_rate = acc_cnt*1.0/len(s)*100
        if self.learn_step_counter % 100 == 0:
            # print("self.DECAY_PARAMS:{}".format(self.DECAY_PARAMS))
            print("test acc_rate:{}".format(acc_rate))
            print("lr:{}".format(self.lr))
            print("loss:{}".format(cost))
        self.learn_step_counter += 1
        return cost,acc_rate
    def get_prob_eval(self,s):
        q_eval = self.sess.run([self.q_eval], feed_dict={self.s:s,
                                                                self.is_training:False})
        q_eval = q_eval[0]
        return q_eval

    def choose_action(self, s, command_size, epsilon=0,enjoy=False,dead_e_index=None):
        s = np.array(s).reshape([1,self.resolution[0],self.resolution[1], 1])
        if enjoy:
            mq = self.sess.run(self.q_eval, feed_dict={self.s: s,self.is_training:False})[0]
            best_a = np.argmax(mq)
            if dead_e_index == None:
                return best_a
            while True:
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index:
                    # invalid action
                    mq[best_a] = -200000
                    best_a = np.argmax(mq)
                    index = best_a - 9
                else:
                    break
            return best_a


        if random() <= epsilon:
            probs = np.zeros(command_size,dtype=float) + 1.0/command_size
            best_a =  np.random.choice(np.arange(probs.shape[0]), p=probs.ravel())
            if dead_e_index == None:
                return best_a
            while True:
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index:
                    nonzero_cnt = len(np.nonzero(probs)[0]) - 1
                    fix_prob= probs[best_a]
                    probs[best_a]=0
                    for i in range(len(probs)):
                        if probs[i] > 0:
                            probs[i] = probs[i] + fix_prob/nonzero_cnt
                    best_a =  np.random.choice(np.arange(probs.shape[0]), p=probs.ravel())
                    index = best_a - 9

                else:
                    break

        else:
        # forward feed the observation and get q value for every actions
            mq = self.sess.run(self.q_eval, feed_dict={self.s: s,self.is_training:False})[0]
            best_a = np.argmax(mq)
            if dead_e_index == None:
                return best_a
            while True:
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index:
                    # invalid action
                    # print("action:{} invalid action!!!!".format(best_a))
                    mq[best_a] = -200000
                    best_a = np.argmax(mq)
                    index = best_a - 9
                    # print("index:",index)
                else:
                    break
        
        return best_a

class DQN_normal(object):
    """docstring for DQN"""
    def __init__(self, session, resolution=(10,14), command_size = 14, learning_rate=5e-3, buffer_size=10000, replace_target_iter=100,gamma=0.99,batch_size=64, index=0, deeper=False):
        super(DQN_normal, self).__init__()
        self.hidden1_units = 100
        self.hidden2_units = 100
        self.hidden3_units = 100
        self.hidden4_units = 100
        self.lr = learning_rate
        self.sess = session
        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0
        self.gamma = gamma
        self.index = index
        self.resolution = resolution
        self.command_size = command_size
        self._build_net()
        print("DQN_normal!!!!!!!!!!!!!!!!!!!!!")
        print([v.name for v in self.t_params]) #=> prints lists of vars created
        print([v.name for v in self.e_params]) #=> prints lists of vars created
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        self.saver = tf.train.Saver(max_to_keep=CONFIG.max_to_keep)
        self.sess.run(tf.global_variables_initializer())
        
        

        
    def _build_net(self):
        def fully_connected(prev_layer, num_units, is_training):
            layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
            layer = tf.nn.relu(layer)
            return layer
        def conv2d(prev_layer, filters, kernel_size,strides, pool_size,pool_stride,is_training):
            conv = tf.layers.conv2d(   
                inputs=prev_layer,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=tf.nn.relu
            )   
            # if pool_size == -1:  
            #     pass
            # else:   
            #     conv = tf.layers.max_pooling2d(
            #         conv,
            #         pool_size=pool_size,
            #         strides=pool_stride,
            #     )    

            return conv  

        def build_layers(scope, s):
            print("begin build_layers scope:{} .....".format(scope))
            with tf.variable_scope(scope+str(self.index)):
                # conv0 = conv2d(s,filters=32,kernel_size=16,strides=4,pool_size=2,pool_stride=2,is_training=self.is_training)
                # shape (64, 64, 1)
                print("after build conv0")
                conv1 = conv2d(s,filters=32,kernel_size=8,strides=4,pool_size=2,pool_stride=2,is_training=self.is_training)
                # conv1 (18,18,16)
                print("after build conv1")
                conv2 = conv2d(conv1,filters=64,kernel_size=4,strides=2,pool_size=2,pool_stride=2,is_training=self.is_training)
                # conv2 (13,13,32)
                print("after build conv2")
                conv3 = conv2d(conv2,filters=64,kernel_size=3,strides=1,pool_size=2,pool_stride=2,is_training=self.is_training)
                print("after build conv3")
                flatten_view = tf.contrib.layers.flatten(conv3)
                fc1 = fully_connected(flatten_view, num_units=128, is_training=self.is_training)
                fc2 = fully_connected(fc1, num_units=128, is_training=self.is_training)
                fc3 = fully_connected(fc2, num_units=64, is_training=self.is_training)

                q = tf.layers.dense(fc3, self.command_size, use_bias=True, activation=None)
            print("after build_layers scope:{}!".format(scope))
            return q
        with tf.variable_scope('variable'+str(self.index)):
            # self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            self.a = tf.placeholder(tf.int64,[None], name="a")
            
            # ------------------ build evaluate_net ------------------
            self.s = tf.placeholder(tf.float32, [None] + list(self.resolution) + [1], name="State")  # input
            self.q_target = tf.placeholder(tf.float32, [None,self.command_size], name="TargetQ")  # for calculating loss
            # ------------------ build target_net ------------------
            self.s_ = tf.placeholder(tf.float32, [None] + list(self.resolution) + [1], name="State_next")  # input
            
        self.q_eval = build_layers("eval_net",self.s)
        self.q_next = build_layers("target_net",self.s_)
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net'+str(self.index))
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net'+str(self.index))

        with tf.variable_scope('loss'+str(self.index)):

            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval), name="TD_error")

        with tf.variable_scope('train'+str(self.index)):
            self.learning_rate = tf.placeholder(tf.float32)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            self.gradients = tf.gradients(self.loss,self.e_params)
            self._train_op = optimizer.apply_gradients(zip(self.gradients, self.e_params))


    def set_e_params(self,t_params):
        self.target_replace_op_e = [tf.assign(e, t) for e, t in zip(self.e_params, t_params)]
        self.sess.run(self.target_replace_op_e)
    def set_t_params(self,t_params):
        self.target_replace_op_t = [tf.assign(e, t) for e, t in zip(self.t_params, t_params)]
        self.sess.run(self.target_replace_op_t)

    def get_e_params(self):
        return self.sess.run(self.e_params)
    def get_t_params(self):
        return self.sess.run(self.t_params)

    def set_lr(self,learning_rate,lr_decay_rate,min_lr,decay_round):
        self.lr = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.min_lr = min_lr
        self.decay_round = decay_round
    def set_decay_params(self,DECAY_PARAMS):
        self.True_DECAY_PARAMS = DECAY_PARAMS
    def validation(self,s,a, dead_e_index):
        #----------------------mean zero begin---------------------------------
        # print("s.shape:{}".format(np.array(s).shape))
        s=np.array(s).reshape([-1,self.resolution[0],self.resolution[1],1])
        #----------------------mean zero end------------------------------------------
        q_eval = self.sess.run([self.q_eval], feed_dict={self.s:s,
                                                                self.is_training:False})

        q_eval = q_eval[0]

        acc_cnt = 0
        for i in range(0,len(s)):
            best_a = np.argmax(q_eval[i])

            while True:
                if dead_e_index is None:
                    break
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index[i]:
                    
                    q_eval[i][best_a] = -200000
                    best_a = np.argmax(q_eval[i])
                    index = best_a - 9
                    # print("index:",index)
                else:
                    break
            if best_a == a[i]:
                acc_cnt+=1
        acc_rate = acc_cnt*1.0/len(s)*100
        return acc_rate



    def learn_with_one_episode(self,s,a,s_,isterminal,r,command_size,dead_e_index=None,double_dqn=False, t_prob_eval=None,use_old=None):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
        loss_sum = 0
        lr = 0
        target_q = []
        q_eval_list = []
        q_next_list = []
        q_target = self.sess.run(self.q_eval,feed_dict={self.s:s,self.is_training:False})
        if double_dqn:
            q_eval_list = self.sess.run(self.q_eval,feed_dict={self.s:s_,self.is_training:False})

            q_next_list = self.sess.run(self.q_next,feed_dict={self.s_:s_,self.is_training:False})
        else:
            q_next_list = self.sess.run(self.q_next,feed_dict={self.s_:s_,self.is_training:False})

        for i in range(0,len(s)):
            if not bool(isterminal[i]):
                if double_dqn:
                    q_next = q_next_list[i]
                    q_eval = q_eval_list[i]

                    q2 = q_next[ np.argmax(q_eval) ]
                else:

                    q_next = q_next_list[i]
                    q2 = np.max(q_next)
                tmp_target_q = r[i]  +  q2
            else:
                tmp_target_q = r[i] 
            target_q.append(tmp_target_q)

        batch_size = len(s)
        batch_index = np.arange(batch_size,dtype=np.int32)
        q_target[batch_index,a] = target_q 

        cost,_,q_eval = self.sess.run([self.loss,self._train_op, self.q_eval], 
                                        feed_dict={ self.s:s,
                                                    self.q_target:q_target, 
                                                    self.is_training:True,
                                                    self.learning_rate:self.lr})
        acc_rate = 0

        acc_cnt = 0
        for i in range(0,len(s)):
            best_a = np.argmax(q_eval[i])
            while True:
                if dead_e_index is None:
                    break
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index[i]:
                    q_eval[i][best_a] = -200000
                    best_a = np.argmax(q_eval[i])
                    index = best_a - 9
                else:
                    break
            if best_a == a[i]:
                acc_cnt+=1
        acc_rate = acc_cnt*1.0/len(s)*100


        if self.learn_step_counter % 100 == 0:
            print("test acc_rate:{}".format(acc_rate))
            print("lr:{}".format(self.lr))
            print("loss:{}".format(cost))
        self.learn_step_counter += 1
        return cost,acc_rate


    def choose_action(self, s, command_size, epsilon=0,enjoy=False,dead_e_index=None):
        s = np.array(s).reshape([1,self.resolution[0],self.resolution[1], 1])
        if enjoy:
            mq = self.sess.run(self.q_eval, feed_dict={self.s: s,self.is_training:False})[0]
            best_a = np.argmax(mq)
            while True:
                if dead_e_index is None:
                    break
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index:
                    # invalid action
                    # print("action:{} invalid action!!!!".format(best_a))
                    mq[best_a] = -200000
                    best_a = np.argmax(mq)
                    index = best_a - 9
                    # print("index:",index)
                else:
                    break
            return best_a


        if random() <= epsilon:
            probs = np.zeros(command_size,dtype=float) + 1.0/command_size
            best_a =  np.random.choice(np.arange(probs.shape[0]), p=probs.ravel())
            while True:
                if dead_e_index is None:
                    break
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index:
                    nonzero_cnt = len(np.nonzero(probs)[0]) - 1
                    fix_prob= probs[best_a]
                    probs[best_a]=0
                    for i in range(len(probs)):
                        if probs[i] > 0:
                            probs[i] = probs[i] + fix_prob/nonzero_cnt
                    best_a =  np.random.choice(np.arange(probs.shape[0]), p=probs.ravel())
                    index = best_a - 9

                else:
                    break

        else:
        # forward feed the observation and get q value for every actions
            mq = self.sess.run(self.q_eval, feed_dict={self.s: s, self.is_training:False})[0]
            best_a = np.argmax(mq)

            while True:
                if dead_e_index is None:
                    break
                if best_a <= 8:
                    break
                index = best_a - 9
                if index in dead_e_index:
                    # invalid action
                    # print("action:{} invalid action!!!!".format(best_a))
                    mq[best_a] = -200000
                    best_a = np.argmax(mq)
                    index = best_a - 9
                    # print("index:",index)
                else:
                    break
        
        return best_a







        
        
