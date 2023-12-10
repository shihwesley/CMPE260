import tensorflow as tf
import numpy as np
from random import choice, uniform
from collections import deque
import networkx as nx

from cnn import Cnn
from config import LEARNING_RATE, EPSILON_GREEDY_START_PROB, EPSILON_GREEDY_END_PROB, EPSILON_GREEDY_MAX_STATES, \
    MAX_MEM, BATCH_SIZE, VISION_W, VISION_B, VISION_F, TARGET_NETWORK_UPDATE_FREQUENCY, LEARN_START

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

FLAGS = tf.app.flags.FLAGS

V2V_DATA_SIZE = 500


class DeepTrafficAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.action_names = ['A', 'D', 'M', 'L', 'R']
        self.num_actions = len(self.action_names)
        self.memory = deque()

        self.model = Cnn(self.model_name, self.memory)
        self.target_model = Cnn(self.model_name, [], target=True)

        self.v2v_graph = nx.Graph()

        # self.state = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 1])
        self.previous_states = np.zeros([1, VISION_F + VISION_B + 1, VISION_W * 2 + 1, 4])
        self.previous_actions = np.zeros([4])
        self.previous_actions.fill(2)
        self.q_values = np.zeros(5)
        self.action = 2

        self.count_states = self.model.get_count_states()

        self.delay_count = 0

        self.epsilon_linear = LinearControlSignal(start_value=EPSILON_GREEDY_START_PROB,
                                                  end_value=EPSILON_GREEDY_END_PROB,
                                                  repeat=False)

        self.advantage = 0
        self.value = 0

        self.score = 0

    def get_action_name(self, action):
        return self.action_names[action]

    def get_action_index(self, action):
        return self.action_names.index(action)

    def update_v2v_graph(self, main_car_id, nearby_cars_data):
        """
        Update the V2V graph with new data.
        :param main_car_id: Identifier for the main car.
        :param nearby_cars_data: A list of tuples, each containing the ID of a nearby car and its data.
        """
        # Add/update the main car node
        self.v2v_graph.add_node(main_car_id, data='main_car_data')  # Replace 'main_car_data' with actual data

        # Add/update nodes and edges for nearby cars
        for car_id, car_data in nearby_cars_data:
            self.v2v_graph.add_node(car_id, data=car_data)
            self.v2v_graph.add_edge(main_car_id, car_id)

            # Update nodes and edges for each car's nearby cars
            for nearby_car_id, nearby_car_data in car_data['nearby_cars']:
                self.v2v_graph.add_node(nearby_car_id, data=nearby_car_data)
                self.v2v_graph.add_edge(car_id, nearby_car_id)

    def process_v2v_data(self, v2v_data):
        """
        Process V2V data to update the graph.
        :param v2v_data: The V2V data received by the main car.
        """
        # Extract the main car ID and the data of nearby cars
        main_car_id = v2v_data['main_car_id']
        nearby_cars_data = v2v_data['nearby_cars']

        # Update the V2V graph with the new data
        self.update_v2v_graph(main_car_id, nearby_cars_data)

    def act(self, vision_data, v2v_data, is_training=True):
        # Process the vision data
        vision_processed = self.process_vision_data(vision_data)

        # Process the V2V data
        v2v_processed = self.process_v2v_data(v2v_data)

        # Combine the processed vision and V2V data
        combined_state = self.combine_states(vision_processed, v2v_processed)

        # Decision making with the combined state
        # Here, you would use your neural network model to get the Q-values and decide on an action
        # For simplicity, I'm using a placeholder for the Q-values and action decision
        q_values = self.model.predict_q_values(combined_state)
        action = np.argmax(q_values) if not self.should_explore(is_training) else self.random_action()

        # Update the agent's internal state if needed
        self.update_internal_state(action, combined_state)

        return q_values, self.get_action_name(action)

    def process_vision_data(self, vision_data):
        # Implement the processing of vision data
        # Example: Reshape and normalize
        vision_processed = vision_data.reshape(VISION_F + VISION_B + 1, VISION_W * 2 + 1)
        return vision_processed

    # def process_v2v_data(self, v2v_data):
    #     # Implement the processing of V2V data
    #     # Example: Flatten and normalize
    #     v2v_processed = v2v_data.flatten()
    #     return v2v_processed

    def combine_states(self, vision_processed, v2v_processed):
        # Implement the logic to combine vision and V2V data
        # Example: Concatenate the two states
        combined_state = np.concatenate([vision_processed, v2v_processed])
        return combined_state

    def should_explore(self, is_training):
        # Implement exploration strategy (e.g., epsilon-greedy)
        return np.random.rand() < self.epsilon_linear.get_value(iteration=self.model.get_count_states()) if is_training else False

    def random_action(self):
        # Choose a random action
        return np.random.choice(self.num_actions)

    def remember(self, reward, next_vision_data, next_v2v_data, end_episode=False, is_training=True):
        # Process the next vision and V2V data
        next_vision_processed = self.process_vision_data(next_vision_data)
        next_v2v_processed = self.process_v2v_data(next_v2v_data)

        # Combine the processed next vision and V2V data
        next_combined_state = self.combine_states(next_vision_processed, next_v2v_processed)

        # Create a memory tuple (current state, action, reward, next state, end episode flag)
        memory_tuple = (self.current_state, self.action, reward, next_combined_state, end_episode)

        # Add the memory tuple to the agent's memory
        self.add_to_memory(memory_tuple)

        # Update the agent's current state
        self.current_state = next_combined_state

        # Perform additional tasks if the episode has ended
        if end_episode:
            self.reset_internal_state()
            self.score = 0

        # Update the agent's score
        self.score = reward

        # Increase the number of states encountered
        self.count_states = self.model.increase_count_states()

    def add_to_memory(self, memory_tuple):
        # Add a memory tuple to the memory, ensuring the memory size doesn't exceed the limit
        if len(self.memory) > MAX_MEM:
            self.memory.popleft()
        self.memory.append(memory_tuple)

    def reset_internal_state(self):
        # Reset the agent's internal state at the end of an episode
        # Example: Reset previous states and actions
        # This method is a placeholder and should be adapted to your specific needs
        self.current_state = np.zeros_like(self.current_state)
        # Reset other internal states as needed


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
