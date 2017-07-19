# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import NoisyNetDQNConfig
from NoisyNetDQN import NoisyNetDQN
from DQN import DQN

def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def run_NoisyNetDQN(index, env):
    with tf.variable_scope('DQfD_' + str(index)):
        agent = NoisyNetDQN(env, NoisyNetDQNConfig())
        # agent = DQN(env, NoisyNetDQNConfig())
    scores = []
    for e in range(NoisyNetDQNConfig.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            # action = agent.egreedy_action(state)  # e-greedy action for train
            action = agent.noisy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done)
            agent.train_Q_network(update=False)
            state = next_state
        if done:
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            print "episode:", e, "  score:", score, "  memory length:", len(agent.replay_buffer)
            # if np.mean(scores[-min(10, len(scores)):]) > 495:
            #     break
    return scores


if __name__ == '__main__':
    env = gym.make(NoisyNetDQNConfig.ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    dqfd_sum_scores = np.zeros(NoisyNetDQNConfig.episode)
    for i in range(NoisyNetDQNConfig.iteration):
        scores = run_NoisyNetDQN(i, env)
        dqfd_sum_scores = [a + b for a, b in zip(scores, dqfd_sum_scores)]
    dqfd_mean_scores = dqfd_sum_scores / NoisyNetDQNConfig.iteration
    with open('/Users/mahailong/DQfD/dqfd_mean_scores.p', 'wb') as f:
        pickle.dump(dqfd_mean_scores, f, protocol=2)

    # map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
        # xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
    # env.close()
    # gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')


