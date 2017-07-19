# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


class DQN:
    def __init__(self, env, config):
        self.sess = tf.InteractiveSession()
        self.config = config
        # init experience replay
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)
        self.time_step = 0
        self.epsilon = self.config.INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.action_batch = tf.placeholder("int32", [None])
        self.y_input = tf.placeholder("float", [None, self.action_dim])

        self.eval_input = tf.placeholder("float", [None, self.state_dim])
        self.Q_eval
        self.select_input = tf.placeholder("float", [None, self.state_dim])
        self.Q_select

        self.loss
        self.optimize
        self.update_target_net

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        self.save_model()
        self.restore_model()

    def build_layers(self, state, c_names, units_1, units_2, w_i, b_i, reg=None):
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.state_dim, units_1], initializer=w_i, collections=c_names, regularizer=reg)
            b1 = tf.get_variable('b1', [1, units_1], initializer=b_i, collections=c_names, regularizer=reg)
            dense1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [units_1, units_2], initializer=w_i, collections=c_names, regularizer=reg)
            b2 = tf.get_variable('b2', [1, units_2], initializer=b_i, collections=c_names, regularizer=reg)
            dense2 = tf.nn.relu(tf.matmul(dense1, w2) + b2)
        with tf.variable_scope('l3'):
            w3 = tf.get_variable('w3', [units_2, self.action_dim], initializer=w_i, collections=c_names, regularizer=reg)
            b3 = tf.get_variable('b3', [1, self.action_dim], initializer=b_i, collections=c_names, regularizer=reg)
            dense3 = tf.matmul(dense2, w3) + b3
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
        return tf.reduce_mean(tf.squared_difference(self.Q_select, self.y_input))

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
        # epsilon是不断变小的，也就是随机性不断变小:开始需要更多的探索，所以动作偏随机，之后需要动作能够有效，因此减少随机。
        self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)
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

        # 提供给placeholder，因此需要先计算出
        Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: next_state_batch})

        # convert true to 1, false to 0
        done = np.array(done) + 0

        y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        for i in range(0, self.config.BATCH_SIZE):
            temp = self.Q_select.eval(feed_dict={self.select_input: state_batch[i].reshape((-1, self.state_dim))})[0]
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

    def egreedy_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(self.Q_select.eval(feed_dict={self.select_input: [state]})[0])
