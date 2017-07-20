# NoisyNetDQN
#### Tensorflow implementation for "Noisy network for exploration"

##### The Q-value function is implemented with 2 convolution layers and 3 fully connected layers, and I use the atari game Breakout-v0 for the test.

##### If you are doing test on the CartPole or some other games which the state is 1 demotion, there network of Q-value function should only have dense layers.

##### As a comparison, you can see the implementation of DQN in the DQN.py.

##### The feedback after we execute an action contains 4 parts:
    state, real-reward, game_over, lives_rest

##### There are 4 action in the game Breakout-v0:
    0: hold and doing nothing
    1: throw the ball
    2: move right
    3: move left
    
##### The train-reward is equal to the real-reward in most of the time. But in the beginning of the new live, Its obvious the action should be throwing the ball. So the train-reward should be 1 in spite of the real-reward of throwing is zero. And the train-reward is -1 when you miss the ball.

