from collections import deque
from random import choice, uniform

import numpy as np
import tensorflow as tf

from cnn import Cnn
from config import (BATCH_SIZE, EPSILON_GREEDY_END_PROB,
                    EPSILON_GREEDY_MAX_STATES, EPSILON_GREEDY_START_PROB,
                    LEARN_START, LEARNING_RATE, MAX_MEM, MAX_SIMULATION_CAR,
                    SWITCHING_LANE_REWARD, TARGET_NETWORK_UPDATE_FREQUENCY,
                    VISION_B, VISION_F, VISION_W)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

FLAGS = tf.app.flags.FLAGS

GAMMA = 0.99


class DeepTrafficAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.action_names = ['A', 'D', 'M', 'L', 'R']
        self.num_actions = len(self.action_names)
        self.memory = deque()

        self.model = Cnn(self.model_name, self.memory)
        self.target_model = Cnn(self.model_name, [], target=True)

        self.previous_states = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4])
        self.previous_actions = np.zeros([5])
        self.current_v2v_data = np.zeros([1, 1000])
        self.current_state = np.zeros([1, 1000])

        self.previous_actions.fill(2)
        self.action = 2

        self.count_states = self.model.get_count_states()

        self.delay_count = 0

        self.epsilon_linear = LinearControlSignal(start_value=EPSILON_GREEDY_START_PROB,
                                                  end_value=EPSILON_GREEDY_END_PROB,
                                                  repeat=False)

        self.advantage = 0
        self.value = 0

        self.score = 0

        # Implementing V2V
        # Introduce an attribute for V2V data, assuming a fixed size (e.g., 10)
        self.v2v_data_size = 1000
        self.previous_v2v_data = np.zeros([1, self.v2v_data_size])

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def reset_episode_data(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    @staticmethod
    def extract_ego_car_info(vision):
        print("Vision data:", vision)
        # Assuming VISION_W and VISION_F are defined in your config.py
        ego_car_x = VISION_W  # X-coordinate of the ego car in the center of the vision
        ego_car_y = VISION_F  # Y-coordinate of the ego car in the center of the vision

        ego_car_x = int(ego_car_x)
        ego_car_y = int(ego_car_y)

        # Extract ego car's speed, lane, or any other relevant information from vision
        ego_car_speed = vision[ego_car_y][ego_car_x]['speed']
        ego_car_lane = vision[ego_car_y][ego_car_x]['lane']

        return ego_car_x, ego_car_y, ego_car_speed, ego_car_lane

    def preprocess_v2v_data(self, v2v_data, fixed_size=500):
        # Assume max_cars is the maximum number of cars we consider
        max_cars = 60
        num_features = 5  # Example: x, y, speed, lane, action

        # Calculate the required size
        required_size = max_cars * num_features

        # Initialize an array to hold the processed V2V data
        processed_data = np.zeros((max_cars, num_features))

        for i, car_data in enumerate(v2v_data[:max_cars]):  # Limit to max_cars
            if 'x' in car_data and 'y' in car_data and 'speed' in car_data and 'lane' in car_data and 'intended_action' in car_data:
                processed_data[i] = [car_data['x'], car_data['y'], car_data['speed'], car_data['lane'], self.action_to_int(car_data['intended_action'])]
            # Else block for missing data can be handled as needed

        # Flatten the processed data
        processed_data_flat = processed_data.flatten()

        # Resize to fixed size
        if required_size < fixed_size:
            # Pad with zeros if necessary
            processed_data_flat = np.pad(processed_data_flat, (0, fixed_size - required_size), 'constant')
        elif required_size > fixed_size:
            # Truncate if necessary
            processed_data_flat = processed_data_flat[:fixed_size]

        # Reshape to ensure it's always a 2D array
        processed_data_reshaped = processed_data_flat.reshape(1, -1)

        return processed_data_reshaped

    @staticmethod
    def action_to_int(action):
        # Map intended actions to integer values
        # You can define your own mapping based on your use case
        action_mapping = {'A': 0, 'D': 1, 'M': 2, 'L': 3, 'R': 4}
        return action_mapping.get(action, 2)


    def get_action_name(self, action):
        return self.action_names[action]

    def get_action_index(self, action):
        return self.action_names.index(action)

    def act(self, state, v2v_data, is_training=True):
        # Preprocess V2V data and ensure it is assigned to processed_v2v_data
        processed_v2v_data = self.preprocess_v2v_data(v2v_data)
        num_features = processed_v2v_data.shape[1]

        # Reshape V2V data to match the expected shape
        v2v_data_reshaped = processed_v2v_data.reshape(-1, num_features)

        if v2v_data_reshaped is None:
            # Handle the case where preprocess_v2v_data returns None
            # You might need to initialize processed_v2v_data to some default value
            # depending on how your preprocess_v2v_data function is supposed to work
            v2v_data_reshaped = np.zeros_like(self.previous_v2v_data[:, -1, :])



        # Update previous V2V data
        # Check dimensions before rolling
        if self.previous_v2v_data.shape[1:] != v2v_data_reshaped.shape[1:]:
            # Adjust dimensions if necessary
            self.previous_v2v_data = np.zeros((1, *v2v_data_reshaped.shape[1:]))

        self.previous_v2v_data = np.roll(self.previous_v2v_data, -1, axis=1)
        self.previous_v2v_data[-1, :] = v2v_data_reshaped

        # Process vision data
        state = state.reshape(VISION_F + VISION_B + 1, VISION_W * 2 + 1).tolist()
        previous_states = self.previous_states.tolist()
        for n in range(len(previous_states)):
            for y in range(len(previous_states[n])):
                for x in range(len(previous_states[n][y])):
                    previous_states[n][y][x].pop(0)
                    previous_states[n][y][x].append(state[y][x])
        self.previous_states = np.array(previous_states, dtype=int)
        self.previous_states = self.previous_states.reshape(1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4)

        # Update actions
        self.previous_actions = np.roll(self.previous_actions, -1)
        self.previous_actions[-1] = self.action

        # Combine vision and V2V data for model input
        #combined_input = np.concatenate((self.previous_states.reshape(1, -1), self.previous_v2v_data), axis=1)

        # Get Q-values from the model
        self.previous_actions = self.previous_actions.reshape(1, -1)
        self.q_values = self.model.get_q_values(self.previous_states, self.previous_v2v_data, self.previous_actions)
        self.q_values = self.q_values[0][0]

        if is_training and self.epsilon_linear.get_value(iteration=self.model.get_count_states()) > uniform(0, 1):
            # or (not is_training and EPSILON_GREEDY_TEST_PROB > uniform(0, 1)):
            self.action = choice([0, 1, 2, 3, 4])
        else:
            self.action = np.argmax(self.q_values)

        return self.q_values, self.get_action_name(self.action)

    def remember(self, current_state, current_v2v_data, reward, next_state, next_v2v_data, end_episode=False, is_training=True):
        processed_next_v2v_data = self.preprocess_v2v_data(next_v2v_data)
        self.previous_v2v_data = processed_next_v2v_data.reshape(1, -1)  # Update the V2V data

        processed_current_v2v_data = self.preprocess_v2v_data(current_v2v_data)
        #self.previous_current_v2v_data = processed_current_v2v_data.reshape(1, -1)  # Update the V2V data
        combined_state = np.concatenate((current_state.reshape(1, -1), processed_current_v2v_data.reshape(1, -1)), axis=1)
        combined_state = combined_state.reshape(1, -1)

        # Combine vision and V2V data for the next state
        combined_next_state = np.concatenate((next_state.reshape(1, -1), self.previous_v2v_data), axis=1)
        combined_next_state = combined_next_state.reshape(1, -1)

        next_state = next_state.reshape(VISION_F + VISION_B + 1, VISION_W * 2 + 1).tolist()

        previous_states = self.previous_states.tolist()
        for n in range(len(previous_states)):
            for y in range(len(previous_states[n])):
                for x in range(len(previous_states[n][y])):
                    previous_states[n][y][x].pop(0)
                    previous_states[n][y][x].append(next_state[y][x])
        next_state = np.array(previous_states).reshape(-1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4)

        next_actions = self.previous_actions.copy()
        next_actions = np.roll(next_actions, -1)
        next_actions[0] = self.action

        self.count_states = self.model.get_count_states()

        if is_training and self.model.get_count_states() > LEARN_START and len(self.memory) > LEARN_START:
            self.model.save_checkpoint(self.model.get_count_states())
            self.model.optimize(self.memory,
                                learning_rate=LEARNING_RATE,
                                batch_size=BATCH_SIZE,
                                target_network=self.target_model)

            if self.model.get_count_states() % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                self.model.save_checkpoint(self.model.get_count_states())
                self.target_model.load_checkpoint()
                self.model.log_target_network_update()
                print("Target network updated")
            elif self.model.get_count_states() % 1000 == 0:
                self.model.save_checkpoint(self.model.get_count_states())

        reward = reward + SWITCHING_LANE_REWARD if self.action != 2 else reward

        if len(self.memory) > MAX_MEM:
            self.memory.popleft()
        self.memory.append((combined_state, combined_next_state, self.action, reward - self.score, end_episode))


        self.score = reward

        # Store state, action, and reward
        self.episode_states.append(current_state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(reward)

        if end_episode:
            # If episode ended, perform policy update
            self.update_policy()
            self.reset_episode_data()
            self.previous_states = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4])
            self.previous_actions = np.zeros([5])
            self.previous_actions.fill(2)
            self.q_values = np.zeros(5)
            self.action = 2
            self.score = 0

        self.count_states = self.model.increase_count_states()

    def update_policy(self):
        # Calculate returns
        G = np.zeros_like(self.episode_rewards)
        cumulative_reward = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative_reward = cumulative_reward * GAMMA + self.episode_rewards[t]
            G[t] = cumulative_reward

        # Normalize returns
        G -= np.mean(G)
        G /= np.std(G)

        # Update policy
        for state, action, Gt in zip(self.episode_states, self.episode_actions, G):
            self.model.perform_policy_gradient_update(state, action, Gt)



class LinearControlSignal:
    """
    A control signal that changes linearly over time.
    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.

    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, repeat=False):
        """
        Create a new object.
        :param start_value:
            Start-value for the control signal.
        :param end_value:
            End-value for the control signal.
        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.
        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = EPSILON_GREEDY_MAX_STATES
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / self.num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value
