# -*- coding: utf-8 -*-


class NoisyNetDQNConfig:
    ENV_NAME = "CartPole-v1"
    # ENV_NAME = "Breakout-v0"
    # ENV_NAME = "Freeway-v0"
    GAMMA = 0.99  # discount factor for target Q
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 64  # size of minibatch
    UPDATE_TARGET_NET = 400  # update eval_network params every 200 steps
    LEARNING_RATE = 0.01
    MODEL_PATH = './model/NoisyNetDQN_model'

    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILIN_DECAY = 0.999
    DEMO_RATIO = 0.1
    LAMBDA_1 = 1.0
    LAMBDA_2 = 10e-5
    PRETRAIN_STEPS = 1000
    DEMO_DATA_PATH = '/Users/mahailong/DQfD/demo.p'

    replay_buffer_size = 2000
    iteration = 5
    episode = 3009  # 300 games per iteration

    noisy_distribution = 'factorised'  # independent or factorised




