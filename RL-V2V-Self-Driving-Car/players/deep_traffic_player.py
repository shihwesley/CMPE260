from players.player import Player
import numpy as np


class DeepTrafficPlayer(Player):
    def decide_with_vision_and_v2v(self, vision, v2v_data, score, end_episode, cache=False, is_training=True):
        if cache:
            self.car.move(self.action_cache)
            return None, None

        # If it's not the first iteration, remember the previous transition
        if hasattr(self, 'current_state'):
            self.agent.remember(
                current_state=self.current_state,
                current_v2v_data=self.current_v2v_data,
                reward=score,
                next_state=vision,
                next_v2v_data=v2v_data,
                end_episode=end_episode,
                is_training=is_training
            )

        # Update the current state and V2V data
        self.current_state = vision
        self.current_v2v_data = v2v_data

        action = 'M'  # Default action

        if self.car.switching_lane < 0:
            # Updated to include V2V data
            q_values, action = self.agent.act(vision, v2v_data, is_training=is_training)
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
        # Modify the reward to encourage lane-switching
        #reward += SWITCHING_LANE_REWARD  # Add a positive reward for lane-switching

        resulted_action = self.car.move(action)
        if resulted_action != action:
            self.agent.action = self.agent.action_names.index(resulted_action)
        self.action_cache = resulted_action
        result = resulted_direction if not mismatch_direction else resulted_action

        return q_values if self.agent_action else None, \
            result if self.agent_action else None
