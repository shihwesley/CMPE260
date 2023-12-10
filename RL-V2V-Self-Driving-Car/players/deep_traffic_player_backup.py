from players.player import Player
import numpy as np


class DeepTrafficPlayer(Player):

    def decide_with_vision_and_v2v(self, vision, v2v_data, score, end_episode, cache=False, is_training=True):
        if cache:
            self.car.move(self.action_cache)
            return None, None

        # Process the V2V data
        processed_v2v_data = self.process_v2v_data(v2v_data)

        action = 'M'  # Default action
        if is_training and not cache:
            self.agent.remember(score, vision, processed_v2v_data, end_episode=end_episode, is_training=is_training)

        if self.car.switching_lane < 0:
            q_values, action = self.agent.act(vision, processed_v2v_data, is_training=is_training)
            self.agent_action = True
        else:
            self.agent_action = False

        mismatch_direction = False
        resulted_direction = 'M'

        if self.agent_action:
            if action in ['A', 'D', 'M']:
                self.car.switch_lane('M')
                mismatch_direction = True
            else:
                resulted_direction = self.car.switch_lane(action)
                if resulted_direction[0] != action[0]:
                    self.agent.action = self.agent.action_names.index(resulted_direction)
                    mismatch_direction = True
                action = 'M'

            # Update the intended_action attribute
            self.intended_action = action

        resulted_action = self.car.move(action)
        if resulted_action != action:
            self.agent.action = self.agent.action_names.index(resulted_action)
        self.action_cache = resulted_action
        result = resulted_direction if not mismatch_direction else resulted_action

        return q_values if self.agent_action else None, \
            result if self.agent_action else None

    def process_v2v_data(self, v2v_data):
        processed_data = []

        for car_info in v2v_data:
            # Relative positioning
            rel_x = car_info['x'] - self.car.x
            rel_y = car_info['y'] - self.car.y

            # Speed difference
            speed_diff = car_info['speed'] - self.car.speed

            # Encoding intended actions
            action_encoding = self.encode_action(car_info['intended_action'])

            # Append the processed information
            processed_data.append([rel_x, rel_y, speed_diff, action_encoding])

        # Normalize the data
        processed_data = self.normalize_data(processed_data)

        # Flatten or reshape the data to fit the DQN's input layer
        processed_data = self.flatten_or_reshape_data(processed_data)

        return processed_data

    def encode_action(self, action):
        # Map actions to numerical values
        action_map = {'A': 0, 'D': 1, 'M': 2, 'L': 3, 'R': 4}
        return action_map.get(action, -1)  # Default to -1 for unknown actions

    def normalize_data(self, data):
        # Implement normalization logic, e.g., Min-Max scaling or Z-score normalization
        # This is a placeholder for demonstration
        normalized_data = [[(val / 100) for val in row] for row in data]
        return normalized_data

    def flatten_or_reshape_data(self, data):
        # Flatten or reshape the data to fit your DQN's input layer
        # Example: Flatten the data
        flattened_data = [val for sublist in data for val in sublist]
        return flattened_data

