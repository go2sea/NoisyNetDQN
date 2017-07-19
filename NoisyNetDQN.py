# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools

from myUtils import conv, noisy_dense


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


class NoisyNetDQN:
    def __init__(self, env, config):
        self.sess = tf.InteractiveSession()
        self.config = config
        # init experience replay
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)
        self.time_step = 0
        # self.epsilon = self.config.INITIAL_EPSILON
        # self.state_dim = env.observation_space.shape[0]  # 仅仅适用于状态只有一维的情况，如CartPole
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        print 'state_dim:', self.state_dim
        print 'action_dim:', self.action_dim

        self.action_batch = tf.placeholder("int32", [None])
        self.y_input = tf.placeholder("float", [None, self.action_dim])
        batch_shape = [None]
        batch_shape.extend(self.state_dim)
        self.eval_input = tf.placeholder("float", batch_shape)
        self.select_input = tf.placeholder("float", batch_shape)

        self.Q_select
        self.Q_eval

        self.loss
        self.optimize
        self.update_target_net

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        self.save_model()
        self.restore_model()

    def noise_length(self, units):
        result = 0
        temp = [24]
        temp.extend(units)
        temp.append(self.action_dim)
        for i in range(len(temp) - 1):
            result += temp[i] * temp[i+1] + temp[i+1]
        return result

    def build_layers(self, state, c_names, units_1, units_2, w_i, b_i, reg=None):
        with tf.variable_scope('conv1'):
            conv1 = conv(state, [5, 5, 3, 6], [6], [1, 2, 2, 1], w_i, b_i)
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1, [3, 3, 6, 12], [12], [1, 2, 2, 1], w_i, b_i)
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(conv2)
            # 两种reshape写法
            # flatten = tf.reshape(relu5, [-1, np.prod(relu5.get_shape().as_list()[1:])])
            # flatten = tf.reshape(relu5, [-1, np.prod(relu5.shape.as_list()[1:])])
            # print flatten.get_shape()
        with tf.variable_scope('dense1'):
            dense1 = noisy_dense(flatten, units_1, [units_1], c_names, w_i, b_i)
        with tf.variable_scope('dense2'):
            dense2 = noisy_dense(dense1, units_2, [units_2], c_names, w_i, b_i)
        with tf.variable_scope('dense3'):
            dense3 = noisy_dense(dense2, self.action_dim, [self.action_dim], c_names, w_i, b_i)
        return dense3

    @lazy_property
    def Q_select(self):
        with tf.variable_scope('select_net'):
            c_names = ['select_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)  # 注意：只有select网络有l2正则化
            result = self.build_layers(self.select_input, c_names, 24, 24, w_i, b_i, regularizer)
            return result

    @lazy_property
    def Q_eval(self):
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            result = self.build_layers(self.eval_input, c_names, 24, 24, w_i, b_i)
            return result

    @lazy_property
    def loss(self):
        loss = tf.reduce_mean(tf.squared_difference(self.Q_select, self.y_input))
        return loss

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        return optimizer.minimize(self.loss)  # optimizer只更新selese_network中的参数

    @lazy_property
    def update_target_net(self):
        select_params = tf.get_collection('select_net_params')
        eval_params = tf.get_collection('eval_net_params')
        return [tf.assign(e, s) for e, s in zip(eval_params, select_params)]

    def save_model(self):
        print("Model saved in : ", self.saver.save(self.sess, self.config.MODEL_PATH))

    def restore_model(self):
        self.saver.restore(self.sess, self.config.MODEL_PATH)
        print("Model restored.")

    def perceive(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))  # 经验池添加

    def train_Q_network(self, update=True):
        """
        :param update: True means the action "update_target_net" executes outside, and can be ignored in the function
        """
        if len(self.replay_buffer) < self.config.START_TRAINING:
            return
        self.time_step += 1
        # 经验池随机采样minibatch
        minibatch = random.sample(self.replay_buffer, self.config.BATCH_SIZE)

        np.random.shuffle(minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        # 提供给placeholder，因此需要先计算出（注意区分DDQN：此处不需要计算next_state_batch的Q_select）
        Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: next_state_batch})
        Q_select = self.Q_select.eval(feed_dict={self.select_input: state_batch})

        # convert true to 1, false to 0
        done = np.array(done) + 0

        y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        for i in range(0, self.config.BATCH_SIZE):
            # 注意：noisei_input也只是batch中的一个
            temp = Q_select[i]
            action = np.argmax(Q_eval[i])
            temp[action_batch[i]] = reward_batch[i] + (1 - done[i]) * self.config.GAMMA * Q_eval[i][action]
            y_batch[i] = temp

        # 新产生的样本输入
        self.sess.run(self.optimize, feed_dict={
            self.y_input: y_batch,
            self.select_input: state_batch,
            self.action_batch: action_batch
        })
        # 此例中一局步数有限，因此可以外部控制一局结束后update ，update为false时在外部控制
        if update and self.time_step % self.config.UPDATE_TARGET_NET == 0:
            self.sess.run(self.update_target_net)

    def noisy_action(self, state):
        return np.argmax(self.Q_eval.eval(feed_dict={self.eval_input: [state]})[0])
