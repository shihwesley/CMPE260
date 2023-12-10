import numpy as np

from players.player import Player


class StickyPlayer(Player):
    def __init__(self, car, *args, **kwargs):
        super(StickyPlayer, self).__init__(car, *args, **kwargs)
        self.intended_action = 'M'

    def decide(self, end_episode, cache=False):
        action = 'M'

        if self.car.speed > self.max_speed:
            action = 'D'
        elif self.car.speed < self.min_speed:
            if self.car.switching_lane < 0:
                # Check if there's a car behind and potentially a car in front
                if self.car_in_back(): #and self.car_in_front():
                    # Choose a lane to switch to, considering the safety
                    plan_direction = np.random.choice(['R'])
                    if plan_direction not in self.actions:
                    #if self.can_switch_lanes_safely(plan_direction):
                        self.actions[len(self.actions) - 1] = plan_direction
                        self.car.switch_lane(plan_direction)
                        return
            #action = 'A'
        else:
            if not self.car_in_front():
                action = 'A'

        # Store the action as intended_action
        self.intended_action = action

        self.actions.rotate(-1)
        self.actions[len(self.actions) - 1] = action

        self.car.move(action)