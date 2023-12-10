#!/usr/bin/env/ python
"""
q_learner.py
An easy-to-follow script to train, test and evaluate a Q-learning agent on the Mountain Car
problem using the OpenAI Gym. |Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018
"""
"""
we are going to modify the given Q-learning agent to use the Q-iteration algorithm.
The Q-iteration algorithm is a model-free, off-policy, and tabular TD control algorithm.
"""

import collections

import gym
import numpy as np

#MAX_NUM_EPISODES = 500
MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200  #  This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.4  # Learning rate
GAMMA = 0.66  # Discount factor
NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim

# Initialize the performance recording file
with open("performance_Qiteration.txt", "w") as file:
    file.write("Episode,Reward,Average_Reward,Episode_Length\n")


class Q_Iteration(object):

    def __init__(self, env):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS  # Number of bins to Discretize each observation dim
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        # Create a multi-dimensional array (aka. Table) to represent the
        # Q-values
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,
                           self.action_shape))  # (51 x 51 x 3)
        #we are going to add transition and reward probabilities to the Q-iteration algorithm
        self.transitions = collections.defaultdict(collections.Counter)
        self.recorded_rewards = collections.defaultdict(float)
        #below is given
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def discretize(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        scaled = (obs - self.obs_low) / self.bin_width
        return tuple(scaled.astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def play_n_random_steps(self, count):
        state = self.env.reset()
        for _ in range(count):
            action = self.env.action_space.sample()
            next_state, reward, done, info, _ = self.env.step(action)
            discretized_state = self.discretize(state)
            discretized_next_state = self.discretize(next_state)
            self.transitions[(discretized_state,
                              action)][discretized_next_state] += 1
            self.recorded_rewards[(discretized_state, action,
                                   discretized_next_state)] = reward
            if done:
                state = self.env.reset()
            else:
                state = next_state

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)

        self.transitions[(discretized_obs, action)][discretized_next_obs] += 1
        self.recorded_rewards[(discretized_obs, action,
                               discretized_next_obs)] = reward

        total_transitions = sum(self.transitions[(discretized_obs,
                                                  action)].values())
        q_value_update = 0.0

        for tgt_state, count in self.transitions[(discretized_obs,
                                                  action)].items():
            reward = self.recorded_rewards[(discretized_obs, action, tgt_state)]
            best_next_action = np.argmax(self.Q[tgt_state])
            q_value_update += (count / total_transitions) * (
                reward + GAMMA * self.Q[tgt_state][best_next_action])

        self.Q[discretized_obs][action] = (1 - self.alpha) * self.Q[
            discretized_obs][action] + self.alpha * q_value_update


def train(agent, env):
    agent.play_n_random_steps(10000)
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            total_reward += reward
            obs = next_obs
        if agent.epsilon > EPSILON_MIN:
            agent.epsilon *= EPSILON_DECAY
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(
            episode, total_reward, best_reward, agent.epsilon))
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        with open("performance_Qiteration.txt", "a") as file:
            file.write(
                f"{episode},{total_reward},{best_reward},{agent.epsilon}\n")
    return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, _, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = Q_Iteration(env)
    learned_policy = train(agent, env)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    # for _ in range(1000):
    #     test(agent, env, learned_policy)
    total_test_reward = test(agent, env, learned_policy)
    print("Total test reward:", total_test_reward)
    env.close()
