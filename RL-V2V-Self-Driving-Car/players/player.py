import math
from collections import deque

import numpy as np


class Player:
    def __init__(self, car, min_speed_range=(50, 100), agent=None):
        self.car = car
        self.min_speed = np.random.randint(min_speed_range[0], min_speed_range[1])
        self.max_speed = min_speed_range[1]
        self.agent = agent

        self.action_cache = 'M'
        self.agent_action = False

        self.actions = deque(['M', 'M', 'M', 'M'])
        self.intended_action = 'M'

    def decide(self, end_episode, cache=False):
        # Move forward
        if self.car.speed < self.min_speed:
            action = 'A'
        elif self.car.speed > self.max_speed:
            action = 'D'
        else:
            action = 'M'

        # Update the intended_action attribute
        self.intended_action = action

        self.actions.rotate(-1)
        self.actions[len(self.actions) - 1] = action

        self.car.move(action)

        # Direction
        self.car.switch_lane('M')

    def decide_with_vision_and_v2v(self, vision, v2v_data, score, end_episode, cache=False, is_training=True):
        pass

    def car_in_front(self, threshold=7):
        max_box = min(int(math.floor(self.car.y / 10.0)) - 1, 99)  # Ensure it's not greater than grid size
        min_box = max(max_box - threshold, 0)  # Ensure it's not less than 0

        for y in range(min_box, max_box + 1):
            # Ensure lane index is within range [0, 4] for 5 lanes
            lane_index = self.car.lane - 1
            if 0 <= lane_index < 7:
                if self.car.lane_map[y][lane_index] != 0 and self.car.lane_map[y][lane_index] != self.car:
                    return True

        return False

    def car_in_back(self, threshold=15):
        min_box = max(int(math.floor(self.car.y / 10.0)) + 1, 0)  # Start checking just behind the car
        max_box = min(min_box + threshold, 99)  # Ensure it's not greater than grid size

        for y in range(min_box, max_box + 1):
            # Ensure lane index is within range [0, 4] for 5 lanes
            lane_index = self.car.lane - 1
            if 0 <= lane_index < 7:  # Correct number of lanes to 7
                if self.car.lane_map[y][lane_index] != 0 and self.car.lane_map[y][lane_index] != self.car:
                    return True

        return False

