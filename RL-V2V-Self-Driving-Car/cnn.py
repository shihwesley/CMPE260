import os
import random

import numpy as np
import tensorflow as tf

from config import (DL_IS_TRAINING, ROUND, V2V_DATA_SIZE, VISION_B, VISION_F,
                    VISION_W)

checkpoint_dir = 'models'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

GAMMA = 0.95

class Cnn:
    def __init__(self, model_name, replay_memory, num_actions=5, target=False):
        self.main = not target
        self.model_name = model_name
        self.replay_memory = replay_memory

        self.actions = tf.placeholder(tf.float32, [None, num_actions], name='actions')

        # Define the path for saving/restoring checkpoints
        if self.main:
            model_path = os.path.join(checkpoint_dir, model_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            self.checkpoint_path = os.path.join(model_path, "checkpoint")

        # Define input shapes
        vision_input_shape = [None, VISION_B + VISION_F + 1, VISION_W * 2 + 1, 4]
        v2v_input_shape = [None, V2V_DATA_SIZE]

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Initialize variables
            activation = tf.nn.relu
            init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

            # Placeholders
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
            self.q_values_new = tf.placeholder(tf.float32, shape=[None, num_actions], name='q_values_new')
            self.count_episodes = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name='count_episodes')
            self.count_states = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name='count_states')
            self.count_episodes_increase = tf.assign(self.count_episodes, self.count_episodes + 1)
            self.count_states_increase = tf.assign(self.count_states, self.count_states + 1)
            self.x_vision = tf.placeholder(dtype=tf.float32, shape=vision_input_shape, name='x_vision')
            self.x_v2v = tf.placeholder(dtype=tf.float32, shape=[None, 500], name='x_v2v')
            #self.x = tf.placeholder(tf.float32, [None, input_height, input_width, input_channels])

            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, num_actions])

            # Building layers
            net_vision = self.build_vision_layers(self.x_vision)
            net_v2v = self.build_v2v_layers(self.x_v2v)

            # Combining vision and V2V layers
            net_combined = tf.concat([net_vision, net_v2v], axis=1)

            # Final dense layers
            net_combined = tf.layers.dense(inputs=net_combined, units=100, activation=tf.nn.relu)
            net_combined = tf.layers.dense(inputs=net_combined, units=num_actions, activation=tf.nn.relu)

            self.q_values = net_combined

            # Loss function
            squared_error = tf.square(self.q_values - self.q_values_new)
            sum_squared_error = tf.reduce_sum(squared_error, axis=1)
            self.loss = tf.reduce_mean(sum_squared_error)

            # Optimizer
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # Saver for checkpoints
            self.saver = tf.train.Saver()
            self.init_graph = tf.global_variables_initializer()

        # TensorFlow session
        self.session = tf.Session(graph=self.graph)

        # Load checkpoint or initialize variables
        self.load_checkpoint()
        self.save_checkpoint(self.get_count_states())

        # Setup for logging and TensorBoard
        if self.main and not target:
            log_path = "log/round{}".format(ROUND)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            model_log_path = os.path.join(log_path, model_name)
            if not os.path.exists(model_log_path):
                os.makedirs(model_log_path)
            self.writer = tf.summary.FileWriter(model_log_path, graph=self.session.graph)
            self.writer.flush()

# Reminder: You'll also need to implement build_vision_layers and build_v2v_layers methods.

    def build_vision_layers(self, input_layer):
        # Define activation function and kernel initializer
        activation = tf.nn.relu
        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

        # Conv1
        net = tf.pad(input_layer, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        net = tf.layers.conv2d(
            inputs=net,
            filters=16,
            kernel_size=[3, 3],
            padding='valid',
            activation=activation,
            kernel_initializer=init,
            name='conv1', reuse=None)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, padding='same')

        # Conv2
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        net = tf.layers.conv2d(
            inputs=net,
            filters=32,
            kernel_size=[3, 3],
            padding="valid",
            activation=activation,
            kernel_initializer=init,
            name="conv2", reuse=None)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, padding='same')
        net = tf.reshape(net, [-1, 9 * 1 * 32])

        # Flatten the output of the last convolutional layer
        net = tf.layers.flatten(net)

        return net


    def build_v2v_layers(self, input_layer):
        # Example V2V processing layers
        net = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu)
        net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
        return net

    def close(self):
        """Close the TensorFlow session."""
        self.session.close()

    def load_checkpoint(self):
        """
        Load all variables of the TensorFlow graph from a checkpoint.
        If the checkpoint does not exist, then initialize all variables.
        """
        if not DL_IS_TRAINING or not os.path.exists("{}/{}".format(checkpoint_dir, self.model_name)):
            print("Skipping checkpoint loading for initial training.")
            self.session.run(self.init_graph)
            return

        try:
            print("Trying to restore last checkpoint ...")
            # Use TensorFlow to find the latest checkpoint - if any.
            ckpt = tf.train.latest_checkpoint(checkpoint_dir='{}/{}'.format(checkpoint_dir, self.model_name))

            # Try and load the data in the checkpoint.
            self.saver.restore(self.session, save_path=ckpt)
            print("Restored checkpoint from:", ckpt)

        except Exception as e:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", checkpoint_dir)
            print("Initializing variables instead.")
            self.session.run(self.init_graph)


    def save_checkpoint(self, current_iteration):
        if not self.main or not DL_IS_TRAINING:
            return False
        """Save all variables of the TensorFlow graph to a checkpoint."""
        print("Save checkpoint")
        self.saver.save(self.session,
                        save_path=self.checkpoint_path,
                        global_step=current_iteration)

        try:
            temp_checkpoint_path = os.path.join('{}/{}'.format(checkpoint_dir, self.model_name),
                                                "checkpoint-episode-{}".format(self.get_count_episodes()))

            self.saver.save(self.session,
                            save_path=temp_checkpoint_path,
                            global_step=self.get_count_episodes())
        except Exception as e:
            print(e)

        print("Saved checkpoint.")

    def perform_policy_gradient_update(self, states, actions, Gt):
        with self.graph.as_default():
            # Convert actions to one-hot encoding
            actions_one_hot = tf.one_hot(actions, self.num_actions)

            # Calculate the log probabilities of the taken actions
            log_prob = tf.log(tf.reduce_sum(self.q_values * actions_one_hot, axis=1))

            # Calculate the policy gradient loss
            loss_policy_gradient = -tf.reduce_mean(log_prob * Gt)

            # Create an optimizer and define the training operation
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss_policy_gradient)

            # Run the training operation
            self.session.run(train_op, feed_dict={
                self.x_vision: states['vision'],
                self.x_v2v: states['v2v'],
                self.actions: actions,
                self.Gt: Gt  # You might need to add a placeholder for Gt
            })


    def get_q_values(self, vision_states, v2v_states, actions):
        """
        Calculate and return the estimated Q-values for the given states.
        A single state contains two images (or channels): The most recent
        image-frame from the game-environment, and a motion-tracing image.
        See the MotionTracer-class for details.
        The input to this function is an array of such states which allows
        for batch-processing of the states. So the input is a 4-dim
        array with shape: [batch, height, width, state_channels].

        The output of this function is an array of Q-value-arrays.
        There is a Q-value for each possible action in the game-environment.
        So the output is a 2-dim array with shape: [batch, num_actions]
        """

        # Use the appropriate placeholders for each type of input
        feed_dict = {
            self.x_vision: vision_states,
            self.x_v2v: v2v_states,
            # ... other necessary placeholders ...
            self.actions: actions
        }

        values = self.session.run(self.q_values, feed_dict=feed_dict)

        return values

    def optimize(self, memory, batch_size=128, learning_rate=1e-3, target_network=None):
        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)

        states, vision_states, targets, actions = self.get_memory_component(memory, batch_size, target_network=target_network)

        feed_dict = {self.x_vision: states,
                     self.x_v2v: vision_states,
                     self.actions: actions,
                     self.q_values_new: targets,
                     self.learning_rate: learning_rate}

        # Perform one optimization step and get the loss-value.
        loss_val, _ = self.session.run([self.loss, self.optimizer],
                                       feed_dict=feed_dict)

        # Shift the loss-history and assign the new value.
        # This causes the loss-history to only hold the most recent values.
        loss_history = np.roll(loss_history, 1)
        loss_history[0] = loss_val

        self.log_training_loss(np.average(loss_history))

    def get_memory_component(self, memory, batch_size, target_network=None):
        minibatch = random.sample(memory, batch_size)
        vision_states = []
        v2v_states = []
        actions = []
        targets = []

        for index, (state, next_state, action, reward, end_episode, _actions, next_actions, v2v_state, next_v2v_state) in enumerate(minibatch):
            vision_states.append(state)
            v2v_states.append(v2v_state)
            actions.append(action)
            target = reward

            if not end_episode:
                if target_network:
                    q_values = target_network.get_q_values(next_state, next_v2v_state, next_actions)
                else:
                    q_values = self.get_q_values(next_state, next_v2v_state, next_actions)

                target = reward + GAMMA * np.max(q_values)

            current = self.get_q_values(state, v2v_state, _actions)
            current[0, action] = target  # Assuming action is an integer
            targets.append(current)

        vision_states = np.vstack(vision_states)  # Stack the vision states into a single array
        v2v_states = np.vstack(v2v_states)  # Stack the V2V states into a single array
        targets = np.vstack(targets)  # Stack the targets into a single array
        actions = np.array(actions)  # Convert actions to a numpy array

        return vision_states, v2v_states, targets, actions

    def get_weights_variable(self, layer_name):
        """
        Return the variable inside the TensorFlow graph for the weights
        in the layer with the given name.
        Note that the actual values of the variables are not returned,
        you must use the function get_variable_value() for that.
        """
        # The tf.layers API uses this name for the weights in a conv-layer.
        variable_name = 'kernel'

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable

    def get_variable_value(self, variable):
        """Return the value of a variable inside the TensorFlow graph."""

        weights = self.session.run(variable)

        return weights

    def get_tensor_value(self, tensor, state):
        """Get the value of a tensor in the Neural Network."""

        # Create a feed-dict for inputting the state to the Neural Network.
        feed_dict = {self.x: [state]}

        # Run the TensorFlow session to calculate the value of the tensor.
        output = self.session.run(tensor, feed_dict=feed_dict)

        return output

    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.session.run(self.count_episodes)

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_episodes_increase)

    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. It is just a hack to save and
        reload the counter along with the checkpoint-file.
        """
        return self.session.run(self.count_states)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_states_increase)

    def log_average_speed(self, speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='avg_speed', simple_value=speed)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_testing_speed(self, speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_speed', simple_value=speed)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_training_loss(self, loss):
        summary = tf.Summary(value=[tf.Summary.Value(tag='losses', simple_value=loss)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_total_frame(self, frame):
        summary = tf.Summary(value=[tf.Summary.Value(tag='frames', simple_value=frame)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_terminated(self, terminated):
        summary = tf.Summary(value=[tf.Summary.Value(tag='terminated', simple_value=1 if terminated else 0)])
        self.writer.add_summary(summary, self.get_count_episodes())

    def log_reward(self, reward):
        summary = tf.Summary(value=[tf.Summary.Value(tag='rewards', simple_value=reward)])
        self.writer.add_summary(summary, self.get_count_episodes())

    # def log_average_test_speed(self, test_speed):
    #     summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed', simple_value=test_speed)])
    #     self.writer.add_summary(summary, 0)
    #     self.writer.add_summary(summary, 1)

    def log_average_test_speed_20(self, test_speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed_20', simple_value=test_speed)])
        self.writer.add_summary(summary, 0)
        self.writer.add_summary(summary, 1)

    def log_average_test_speed_40(self, test_speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed_40', simple_value=test_speed)])
        self.writer.add_summary(summary, 0)
        self.writer.add_summary(summary, 1)

    def log_average_test_speed_60(self, test_speed):
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_average_speed_60', simple_value=test_speed)])
        self.writer.add_summary(summary, 0)
        self.writer.add_summary(summary, 1)

    def log_target_network_update(self):
        summary = tf.Summary(value=[tf.Summary.Value(tag='target_update', simple_value=1)])
        self.writer.add_summary(summary, self.get_count_states())

    def log_q_values(self, q_values):
        summary = tf.Summary(value=[tf.Summary.Value(tag='sum_q_values', simple_value=q_values)])
        self.writer.add_summary(summary, self.get_count_states())

    def log_hard_brake_count(self, count):
        summary = tf.Summary(value=[tf.Summary.Value(tag='hard_brake_count', simple_value=count)])
        self.writer.add_summary(summary, self.get_count_states())

    def log_action_frequency(self, stats):
        sum = float(np.sum(stats))
        s = stats.tolist()
        for index, value in enumerate(s):
            summary = tf.Summary(value=[tf.Summary.Value(tag='action_frequency', simple_value=value/sum)])
            self.writer.add_summary(summary, index)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
