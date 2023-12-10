import math
import os

import numpy as np
import pygame

from config import (EMERGENCY_BRAKE_MAX_SPEED_DIFF, ROAD_VIEW_OFFSET, VISION_B,
                    VISION_F, VISION_W, VISUAL_VISION_B, VISUAL_VISION_F,
                    VISUAL_VISION_W, VISUALENABLED)
from players.aggresive_player import AggresivePlayer
from players.deep_traffic_player import DeepTrafficPlayer
from players.player import Player
from players.sticky_player import StickyPlayer

MAX_SPEED = 110  # km/h
MAX_CARS = 10

DEFAULT_CAR_POS = 700

IMAGE_PATH = './images'

if VISUALENABLED:
    red_car = pygame.image.load(os.path.join(IMAGE_PATH, 'red_car.png'))
    red_car = pygame.transform.scale(red_car, (34, 70))
    white_car = pygame.image.load(os.path.join(IMAGE_PATH, 'white_car.png'))
    white_car = pygame.transform.scale(white_car, (34, 70))

direction_weight = {
    'L': 0.2,
    'M': 0.6,
    'R': 0.2,
}

move_weight = {
    'A': 0.30,
    'M': 0.50,
    'D': 0.20
}


class Car():
    def __init__(self, surface, lane_map, speed=0, y=0, lane=2, is_subject=False, subject=None, score=None, agent=None, id=None):
        self.surface = surface
        self.lane_map = lane_map
        self.sprite = None if not VISUALENABLED else red_car if is_subject else white_car
        self.speed = min(max(speed, 0), MAX_SPEED)
        self.y = y
        self.lane = int(lane)
        self.x = (self.lane - 1) * 50 + 15 + 8 + ROAD_VIEW_OFFSET
        self.is_subject = is_subject
        self.subject = subject
        self.max_speed = -1
        self.removed = False
        self.emergency_brake = None

        self.switching_lane = -1
        self.available_directions = ['M']
        self.available_moves = ['D']
        self.nearby_cars = []

        #self.id = id if id is not None else generate_unique_id()  # Implement generate_unique_id as needed

        self.score = score

        self.player = np.random.choice([
                Player(self),
                AggresivePlayer(self),
                StickyPlayer(self)
            ]) if not self.is_subject else DeepTrafficPlayer(self, agent=agent)

        self.hard_brake_count = 0
        self.alternate_line_switching = 0

        last_id = 0

    # def generate_unique_id():
    #     global last_id
    #     last_id += 1
    #     return last_id

    def __repr__(self):
        return f"Car(x={self.x}, y={self.y}, speed={self.speed}, lane={self.lane}, intended_action={self.player.intended_action})"

    def calculate_dynamic_radius(self):
        base_radius = 1000  # Base radius that ensures nearby cars are detected
        scaling_factor = 0.6  # Scaling factor to adjust radius based on speed
        return base_radius + self.speed * scaling_factor

    def find_nearby_cars(self, all_cars):
        radius = self.calculate_dynamic_radius()
        nearby_cars = []
        for car in all_cars:
            if car != self and self.is_within_radius(car, radius) and self.is_relevant_lane(car):
                nearby_cars.append(car)
        return nearby_cars

    def is_within_radius(self, other_car, radius):
        if isinstance(other_car, dict):
            # Access x and y as dictionary keys
            dx = self.x - other_car['x']
            dy = self.y - other_car['y']
        else:
            # Access x and y as attributes
            dx = self.x - other_car.x
            dy = self.y - other_car.y

        distance = math.sqrt(dx**2 + dy**2)
        return distance <= radius

    def is_relevant_lane(self, other_car):
        # Check if the other car is in the same or adjacent lane
        return abs(self.lane - other_car.lane) <= 1

    def identify(self):
        min_box = int(math.floor(self.y / 10.0)) - 1
        max_box = int(math.ceil(self.y / 10.0))

        # Out of bound
        if self.y < -200 or self.y > 1200:
            self.removed = True
            return False

        if 0 <= min_box < 100:

            if 1 <= self.lane <= 5:  # Ensure lane is within range
                self.lane_map[min_box][self.lane - 1] = self
            if 1 <= self.switching_lane <= 5:
                self.lane_map[min_box][self.switching_lane - 1] = self
        for i in range(-1, 9):
            if 0 <= max_box + i < 100:
                if 1 <= self.lane <= 5:  # Ensure lane is within range
                    self.lane_map[max_box + i][self.lane - 1] = self
                if 1 <= self.switching_lane <= 5:
                    self.lane_map[max_box + i][self.switching_lane - 1] = self
        return True

    def accelerate(self):
        # If in front has car then cannot accelerate but follow
        self.speed += 1.0 if self.speed < MAX_SPEED else 0.0

    def decelerate(self):
        if self.max_speed > -1:
            self.speed = self.max_speed
        else:
            self.speed -= 1.0 if self.speed > 0 else 0.0

    def check_switch_lane(self):
        if self.switching_lane == -1:
            return
        self.x += (self.switching_lane - self.lane) * 50
        if self.x == ROAD_VIEW_OFFSET + (self.switching_lane - 1) * 50 + 15 + 8:
            self.lane = self.switching_lane
            self.switching_lane = -1

    def move(self, action):
        moves = self.available_moves

        if action not in moves:
            action = moves[0]
            if self.subject is None:
                self.score.action_mismatch_penalty()

        if action == 'A':
            self.accelerate()
        elif action == 'D':
            self.decelerate()

        return action

    def update_nearby_cars(self, all_cars):
        # Find and update the list of nearby cars
        self.nearby_cars = self.find_nearby_cars(all_cars)

    def broadcast_info(self, all_cars):
        # Update the nearby cars list before broadcasting
        self.update_nearby_cars(all_cars)

        # Broadcast information about the car's current state and nearby cars
        broadcast_data = {
            'id': id(self),
            'x': self.x,
            'y': self.y,
            'speed': self.speed,
            'lane': self.lane,
            'intended_action': self.player.intended_action,  # Assuming each player has an intended action
            'nearby_cars': [car.simple_info() for car in self.nearby_cars]
        }
        return broadcast_data

    def simple_info(self):
        return {
            'id': id(self),
            'x': self.x,
            'y': self.y,
            'speed': self.speed,
            'lane': self.lane,
            'intended_action': self.player.intended_action,
            #'nearby_cars': [{'id': id(car), 'x': car.x, 'y': car.y, 'speed': car.speed, 'lane': car.lane} for car in self.nearby_cars]
        }

    def receive_info(self, info):
        """ Process information received from another car """
        updated_count = 0

        # Process the information of the broadcasting car
        if self.update_nearby_cars_info(info):
            updated_count += 1

        # Process the information of nearby cars
        for car_info in info['nearby_cars']:
            if self.update_nearby_cars_info(car_info):
                updated_count += 1

        return updated_count


    def update_nearby_cars_info(self, car_info):
        # Find the car in the nearby cars list based on x, y, and lane
        car = next((car for car in self.nearby_cars if car.x == car_info['x'] and car.y == car_info['y'] and car.lane == car_info['lane']), None)

        if car:
            # Update the car's information
            car.speed = car_info['speed']
            # Update any other relevant attributes if necessary
        else:
            # If the car is not in the list and is within the radius, add its info
            if self.is_within_radius(car_info, self.calculate_dynamic_radius()):
                new_car = Car(surface=None, lane_map=None, speed=car_info['speed'], y=car_info['y'], lane=car_info['lane'], is_subject=False)
                # Set other necessary attributes of new_car as needed
                self.nearby_cars.append(new_car)
                return True
            else:
                return False

    def remove_outdated_info(self):
        # Remove information about cars that are no longer nearby
        self.nearby_cars = [car for car in self.nearby_cars if self.is_within_radius(car, self.calculate_dynamic_radius())]

    def is_nearby(self, car_info):
        # Determine if a car is nearby (within a certain range)
        distance = ((self.x - car_info['x'])**2 + (self.y - car_info['y'])**2)**0.5
        return distance < 10  # Replace with an appropriate range value

    def switch_lane(self, direction):
        directions = self.available_directions
        if direction == 'R':
            if 'R' in directions:
                if self.lane < 7:
                    self.switching_lane = self.lane + 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        if direction == 'L':
            if 'L' in directions:
                if self.lane > 1:
                    self.switching_lane = self.lane - 1
                    self.identify()
                else:
                    if self.subject is None:
                        self.score.action_mismatch_penalty()
                    return 'M'
        return direction

    def identify_available_moves(self):
        self.max_speed = -1
        moves = ['M', 'A', 'D']
        directions = ['M', 'L', 'R']
        if self.switching_lane >= 0:
            directions = ['M']
        if self.lane == 1 and 'L' in directions:
            directions.remove('L')
        if self.lane == 7 and 'R' in directions:
            directions.remove('R')

        max_box = int(math.ceil(self.y / 10.0)) - 1
        # Front checking
        for i in range(-1, 7):
            if 0 <= max_box + i < 100:
                current_lane_index = self.lane - 1
                if self.lane_map[max_box + i][current_lane_index] != 0 and self.lane_map[max_box + i][current_lane_index] != self:
                    car_in_front = self.lane_map[max_box + i][self.lane - 1]
                    if 'A' in moves:
                        moves.remove('A')
                    if car_in_front.speed < self.speed:
                        if 'M' in moves:
                            moves.remove('M')
                        self.emergency_brake = self.speed - car_in_front.speed
                        self.max_speed = car_in_front.speed
                    break
        # Consider car in target switching lane
        for i in range(-1, 7):
            if 0 <= max_box + i < 100:
                if self.switching_lane > 0:
                    if self.lane_map[max_box + i][self.switching_lane - 1] != 0 and self.lane_map[max_box + i][
                                self.switching_lane - 1] != self:
                        if 'A' in moves:
                            moves.remove('A')
                        car_in_front = self.lane_map[max_box + i][self.switching_lane - 1]
                        if car_in_front.speed < self.speed:
                            if 'M' in moves:
                                moves.remove('M')
                            # emergency_brake = self.speed - car_in_front.speed
                            self.max_speed = car_in_front.speed \
                                if self.max_speed == -1 or self.max_speed > car_in_front.speed else self.max_speed

        # Left lane checking
        if 'L' in directions:
            left_lane_index = self.lane - 2
            if 0 <= left_lane_index < 7:  # Ensure the index is within bounds for 5 lanes
                for i in range(0, 9):
                    if 0 <= max_box + i < 100:
                        if self.lane_map[max_box + i][left_lane_index] != 0:
                            directions.remove('L')
                            break

        # Right lane checking
        if 'R' in directions:
            right_lane_index = self.lane  # For 5 lanes, right lane is current lane + 1
            if right_lane_index < 7:  # Ensure the index is within bounds for 5 lanes
                for i in range(0, 9):
                    if 0 <= max_box + i < 100:
                        if self.lane_map[max_box + i][right_lane_index] != 0:
                            directions.remove('R')
                            break

        self.available_moves = moves
        self.available_directions = directions

        return moves, directions

    def random(self):
        moves, directions = self.identify_available_moves()

        ds = np.random.choice(direction_weight.keys(), 3, p=direction_weight.values())
        ms = np.random.choice(move_weight.keys(), 3, p=move_weight.values())
        for d in ds:
            if d in directions:
                self.switch_lane(d)
                break

        for m in ms:
            if m in moves:
                self.move(m)
                break

    def relative_pos_subject(self):
        if self.is_subject:
            if self.emergency_brake is not None and self.emergency_brake > EMERGENCY_BRAKE_MAX_SPEED_DIFF:
                self.score.brake_penalty()
                self.hard_brake_count += 1
            self.emergency_brake = None
            return
        dvdt = self.speed - self.subject.speed
        dmds = dvdt / 3.6
        dbdm = 1.0 / 0.25
        dsdf = 1.0 / 50.0
        dmdf = dmds * dsdf
        dbdf = dbdm * dmdf * 10.0
        self.y = self.y - dbdf

        if DEFAULT_CAR_POS - dbdf <= self.y < DEFAULT_CAR_POS:
            self.score.subtract()
        elif DEFAULT_CAR_POS - dbdf > self.y >= DEFAULT_CAR_POS:
            self.score.add()
        self.score.penalty()

    def get_v2v_data(self):
        """
        Gather and process V2V data from nearby cars.
        """
        # Initialize an empty list to store the processed V2V data
        v2v_data = []
        # Iterate over the nearby cars and process their data
        for car in self.nearby_cars:
            # Calculate relative positions and other metrics as needed
            rel_x = car.x - self.x
            rel_y = car.y - self.y
            speed_diff = car.speed - self.speed
            lane_diff = car.lane - self.lane
            # Risk assessment (this is a simplistic example)
            risk = 0
            if abs(rel_y) < 5 and abs(speed_diff) > 10:
                risk = 1  # Indicates higher risk
            # Determine the intended action of the nearby car
            intended_action = car.player.intended_action if car.player else 'unknown'
            # You can add other relevant metrics to this data structure
            car_data = {
                'rel_x': rel_x,
                'rel_y': rel_y,
                'speed_diff': speed_diff,
                'lane_diff': lane_diff,
                'intended_action': intended_action,
                'risk': risk
            }
            # Append the processed car data to the V2V data list
            v2v_data.append(car_data)
        # Return the processed V2V data
        return v2v_data

    def decide(self, end_episode, cache=False, is_training=True):
        if self.is_subject:
            # Get vision data
            vision = self.get_vision()

            # Get V2V data - modify this to get the actual V2V data
            v2v_data = self.get_v2v_data()

            # Decision-making with both vision and V2V data
            q_values, result = self.player.decide_with_vision_and_v2v(vision,
                                                                      v2v_data,
                                                                      self.score.score,
                                                                      end_episode,
                                                                      cache=cache,
                                                                      is_training=is_training)
            # Check for recent lane switching
            if result == 'L' or result == 'R':
                if (result == 'L' and 4 in self.player.agent.previous_actions) or \
                        (result == 'R' and 3 in self.player.agent.previous_actions):
                    self.score.switching_lane_penalty()
                    self.alternate_line_switching += 1
            return q_values, result
        else:
            return self.player.decide(end_episode, cache=cache)

    def draw(self):
        self.relative_pos_subject()
        self.check_switch_lane()
        if VISUALENABLED:
            self.surface.blit(self.sprite, (self.x, self.y, 34, 70))

    def get_vision(self):
        min_x = min(max(0, self.lane - 1 - VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISION_W), 6)
        input_min_xx = self.lane - 1 - VISION_W
        input_max_xx = self.lane - 1 + VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars_in_vision = set([
            (self.lane_map[y][x].lane - 1, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0])

        vision = np.zeros((100, 7), dtype=np.int)
        for car in cars_in_vision:
            for y in range(7):
                vision[car[1] + y][car[0]] = 1

        # Crop vision from lane_map
        vision = vision[min_y: max_y + 1, min_x: max_x + 1]

        # Add padding if required
        vision = np.pad(vision,
                        ((min_y - input_min_y, input_max_y - max_y), (min_x - input_min_xx, input_max_xx - max_x)),
                        'constant',
                        constant_values=(-1))

        vision = np.reshape(vision, [VISION_F + VISION_B + 1, VISION_W * 2 + 1, 1])
        return vision

    def get_subjective_vision(self):
        min_x = min(max(0, self.lane - 1 - VISUAL_VISION_W), 6)
        max_x = min(max(0, self.lane - 1 + VISUAL_VISION_W), 6)
        input_min_xx = self.lane - 1 - VISUAL_VISION_W
        input_max_xx = self.lane - 1 + VISUAL_VISION_W

        input_min_y = int(math.floor(self.y / 10.0)) - VISUAL_VISION_F
        input_max_y = int(math.floor(self.y / 10.0)) + VISUAL_VISION_B
        min_y = min(max(0, input_min_y), 100)
        max_y = min(max(0, input_max_y), 100)

        cars = [
            (self.lane_map[y][x].lane, int(math.floor(self.lane_map[y][x].y / 10.0)))
            for y in range(min_y, max_y + 1)
            for x in range(min_x, max_x + 1)
            if self.lane_map[y][x] != 0 and self.lane_map[y][x].subject is not None]

        return cars
