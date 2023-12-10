#!/usr/bin/env python3
import argparse
from typing import List, Optional, Tuple

import gym
import numpy as np
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


class Config:
    GAMMA = 0.99
    LEARNING_RATE = 0.001
    ENTROPY_BETA = 0.01
    BATCH_SIZE = 8
    REWARD_STEPS = 10

class PGN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def preprocess(states: List[Tuple[np.ndarray, dict]]) -> torch.Tensor:
    """ Custom preprocessor for states """
    if not states:
        raise ValueError("Received empty states list")

    # Extract and process only the NumPy array part of each state
    np_states = np.array([s[0] for s in states], dtype=np.float32)
    return torch.from_numpy(np_states)


class CustomExperienceSource(ptan.experience.ExperienceSourceFirstLast):
    def __init__(self, env, agent, **kwargs):
        super().__init__(env, agent, **kwargs)
        self.env = env

    def __iter__(self):
        states = self.env.reset()
        agent_states = self.agent.initial_state()
        history_idx = 0

        while True:
            actions, agent_states = self.agent([states], agent_states)
            if actions is None:
                actions = [None]

            new_states, new_agent_states = [], []
            for env, action, agent_state in zip([self.env], actions, agent_states):
                if action is not None:
                    # Unpack all values returned by env.step()
                    step_result = env.step(action)
                    if len(step_result) > 4:  # If more than 4 values are returned
                        next_state, reward, is_done, info = step_result[:4]
                    else:
                        next_state, reward, is_done, info = step_result

                    # If the episode is done, reset the environment
                    if is_done:
                        next_state = env.reset()

                    # Yield the experience
                    print(f"Current State: {states[history_idx]}, Type: {type(states[history_idx])}")
                    print(f"Action: {action}, Type: {type(action)}")
                    print(f"Reward: {reward}, Type: {type(reward)}")
                    print(f"Next State: {next_state}, Type: {type(next_state)}")
                    print(f"Is Done: {is_done}, Type: {type(is_done)}")

                    yield ptan.experience.ExperienceFirstLast(states[history_idx], action, reward, next_state, is_done)
                else:
                    next_state = states[history_idx]
                    is_done = False

                new_states.append(next_state)
                new_agent_states.append(agent_state)

            # Update the states and agent states
            states, agent_states = new_states, new_agent_states
            history_idx += 1


def train(env, net, agent, optimizer, writer, args):
    exp_source = CustomExperienceSource(env, agent, gamma=Config.GAMMA, steps_count=Config.REWARD_STEPS)
    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_states, batch_actions, batch_scales = [], [], []
    for step_idx, exp in enumerate(exp_source):
        # Unpack experience
        next_state, reward, is_done, _ = exp.state, exp.reward, exp.last, exp.info

        batch_states.append(next_state)
        batch_actions.append(int(exp.action))
        batch_scales.append(reward)

        # Handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        # Prepare batch
        states_v = preprocess(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        # Training step
        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        loss_policy_v.backward(retain_graph=True)
        entropy_v = -(F.softmax(logits_v, dim=1) * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        entropy_loss_v.backward()
        optimizer.step()

        # Clear batch data
        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=False, action='store_true', help="Enable mean baseline")
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    writer = SummaryWriter(comment="-cartpole-pg" + "-baseline=%s" % args.baseline)

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    agent = ptan.agent.PolicyAgent(net, preprocessor=preprocess, apply_softmax=True)
    optimizer = optim.Adam(net.parameters(), lr=Config.LEARNING_RATE)

    debug_exp = ptan.experience.ExperienceFirstLast(np.array([0, 0, 0, 0]), 1, 1.0, np.array([0, 0, 0, 0]), False)
    print("Debug ExperienceFirstLast:", debug_exp)
    train(env, net, agent, optimizer, writer, args)
    writer.close()

