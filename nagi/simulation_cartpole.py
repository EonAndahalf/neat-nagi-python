#import random
#import re
from enum import Enum
from typing import List, Tuple

import numpy as np

from nagi.constants import TIME_STEP_IN_MSEC, \
    ACTUATOR_WINDOW, LIF_SPIKE_VOLTAGE, MAX_POINTS_CARTPOLE_SIM
from nagi.lifsnn import LIFSpikingNeuralNetwork
from nagi.neat import Genome
from nagi.cartpole import NagiCartPoleEnv

# Gaussian distribution for observation receptive fields
def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Sigmoid distribution for observation receptive fields
def sigmoid(x,w=1, shift=0):
    return  1 / (1 + np.exp(-w*(x-shift)))

class CartpoleConfigs(Enum):
    """
    Enum values are the sets containing the input cases where the output of the logic gate is 1.
    """

    # Training
    DEFAULT_M = (0.5, False)
    INVERTED_M = (0.5, True)
    DEFAULT_XS = (0.3, False)
    INVERTED_XS = (0.3, True)
    DEFAULT_XL = (0.7, False)
    INVERTED_XL = (0.7, True)
    # Testing
    DEFAULT_S = (0.4, False)
    INVERTED_S = (0.4, True)
    DEFAULT_L = (0.6, False)
    INVERTED_L = (0.6, True)

    @staticmethod
    def get_training_gates():
        return [CartpoleConfigs.DEFAULT_M, CartpoleConfigs.INVERTED_M,
                CartpoleConfigs.DEFAULT_XS, CartpoleConfigs.INVERTED_XS,
                CartpoleConfigs.DEFAULT_XL, CartpoleConfigs.INVERTED_XL]

    @staticmethod
    def get_testing_gates():
        return [CartpoleConfigs.DEFAULT_S, CartpoleConfigs.INVERTED_S,
                CartpoleConfigs.DEFAULT_L, CartpoleConfigs.INVERTED_L]


class CartpoleAgent(object):
    def __init__(self, key: int, spiking_neural_network: LIFSpikingNeuralNetwork):
        self.spiking_neural_network = spiking_neural_network
        self.key = key
        self.left_actuator = 0
        self.right_actuator = 0
        self.action = 0

    def select_action(self):
        if self.left_actuator > self.right_actuator:
            self.action = 0
        elif self.right_actuator > self.left_actuator:
            self.action = 1
        return self.action

    def reset_actuators(self):
        self.left_actuator = 0
        self.right_actuator = 0
        self.action = 0

    @staticmethod
    def create_agent(genome: Genome):
        return CartpoleAgent(genome.key, LIFSpikingNeuralNetwork.create(genome))

    @staticmethod
    def observation2spike(observation):
        return 0

class CartpoleEnvironment(object):
    def __init__(self, high_frequency: int, low_frequency: int, testing=False):
        self.high_frequency = self._generate_spike_frequency(high_frequency)
        self.low_frequency = self._generate_spike_frequency(low_frequency)
        self.env_config_list = CartpoleConfigs.get_testing_gates() if testing else CartpoleConfigs.get_training_gates()
        self.maximum_possible_lifetime = MAX_POINTS_CARTPOLE_SIM * len(self.env_config_list)
        self.minimum_lifetime = 0

    def simulate(self, agent: CartpoleAgent) -> Tuple[int, float]:
        total_reward = 0
        for env_config in self.env_config_list:
            env = NagiCartPoleEnv(lenght=env_config.value[0], inverted=env_config.value[1])
            observation = env.reset()
            agent.reset_actuators()
            left_actuator = []
            right_actuator = []
            for simulation_step in range(MAX_POINTS_CARTPOLE_SIM):
                for time_step in range(simulation_step *  ACTUATOR_WINDOW,
                                       (simulation_step + 1) * ACTUATOR_WINDOW):
                    frequencies = self._get_input_frequencies(time_step, observation)
                    inputs = self._get_input_voltages(time_step, frequencies)
                    agent.spiking_neural_network.set_inputs(inputs)
                    left, right = agent.spiking_neural_network.advance(TIME_STEP_IN_MSEC)
                    if left:
                        left_actuator.append(time_step)
                    if right:
                        right_actuator.append(time_step)

                agent.left_actuator = CartpoleEnvironment.\
                    _count_spikes_within_time_window(time_step, left_actuator)
                agent.right_actuator = CartpoleEnvironment.\
                    _count_spikes_within_time_window(time_step, right_actuator)

                action = agent.select_action()
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            env.close()
        total_reward = (total_reward / self.maximum_possible_lifetime)
        return (agent.key, total_reward)

    def simulate_with_visualization(self, agent: CartpoleAgent) -> \
            Tuple[int, float, dict, dict, int, List[Tuple[int, int]], List[CartpoleConfigs], List[int]]:

        weights = {key: [] for key, _ in agent.spiking_neural_network.get_weights().items()}
        membrane_potentials = {key: [] for key, _ in
                               agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items()}

        actuator_logger = []
        env_config_logger = []
        reward = 0
        total_reward = 0
        count_time_steps = 0
        flip_points_logger = []

        for env_config in self.env_config_list:
            env = NagiCartPoleEnv(lenght=env_config.value[0], inverted=env_config.value[1])
            observation = env.reset()
            agent.reset_actuators()
            left_actuator = []
            right_actuator = []
            env_config_logger.append(env_config)
            config_reward = 0
            for simulation_step in range(MAX_POINTS_CARTPOLE_SIM):
                env.render()
                for time_step in range(simulation_step *  ACTUATOR_WINDOW,
                                       (simulation_step + 1) * ACTUATOR_WINDOW):
                    count_time_steps += 1
                    actuator_logger.append((agent.left_actuator, agent.right_actuator))
                    for key, weight in agent.spiking_neural_network.get_weights().items():
                        weights[key].append(weight)
                    for key, membrane_potential in agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items():
                        membrane_potentials[key].append(membrane_potential)

                    frequencies = self._get_input_frequencies(time_step, observation)
                    inputs = self._get_input_voltages(time_step, frequencies)
                    agent.spiking_neural_network.set_inputs(inputs)
                    left, right = agent.spiking_neural_network.advance(TIME_STEP_IN_MSEC)
                    if left:
                        left_actuator.append(time_step)
                    if right:
                        right_actuator.append(time_step)

                agent.left_actuator = CartpoleEnvironment.\
                    _count_spikes_within_time_window(time_step, left_actuator)
                agent.right_actuator = CartpoleEnvironment.\
                    _count_spikes_within_time_window(time_step, right_actuator)

                action = agent.select_action()
                observation, reward, done, info = env.step(action)
                total_reward += reward
                config_reward += reward
                print(f'Left: {agent.left_actuator}, Right: {agent.right_actuator}')
                if done:
                    flip_points_logger.append(count_time_steps)
                    break
            print( f'Pole length: {env_config.value[0]:.1f}, Inverted action: {env_config.value[1]}, Reward: {int(config_reward)}')
            env.close()

        actuator_logger.append((agent.left_actuator, agent.right_actuator))
        for key, weight in agent.spiking_neural_network.get_weights().items():
            weights[key].append(weight)
        for key, membrane_potential in agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items():
            membrane_potentials[key].append(membrane_potential)

        total_reward = (total_reward / self.maximum_possible_lifetime)
        print("Total reward:", total_reward)

        return (agent.key,
                total_reward,
                weights,
                membrane_potentials,
                count_time_steps,
                actuator_logger,
                env_config_logger,
                flip_points_logger)

    def _get_input_frequencies(self, time_step: int, observation: List[float]) -> List[float]:
        """
        Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
        """
        input_frequencies = []
        lf = self.low_frequency
        hf = self.high_frequency - self.low_frequency

#        print("lf", lf)
#        print("hf", hf)

        neurons_cart_position = [int(hf*sigmoid(observation[0],-2.5,-0.6)+lf),
                                 int(hf*gaussian(observation[0],0,0.4)+lf),
                                 int(hf*sigmoid(observation[0],2.5,0.6)+lf)]
        input_frequencies.extend(neurons_cart_position)

        neurons_cart_velocity = [int(hf*sigmoid(observation[0],-2.5,-0.6)+lf),
                                 int(hf*gaussian(observation[0],0,0.4)+lf),
                                 int(hf*sigmoid(observation[0],2.5,0.6)+lf)]
        input_frequencies.extend(neurons_cart_velocity)

        neurons_pole_angle = [int(hf*sigmoid(observation[0],-18,-0.1)+lf),
                              int(hf*gaussian(observation[0],0,0.1)+lf),
                              int(hf*sigmoid(observation[0],18,0.1)+lf)]
        input_frequencies.extend(neurons_pole_angle)

        neurons_pole_velocity = [int(hf*sigmoid(observation[0],-2.5,-0.6)+lf),
                                 int(hf*gaussian(observation[0],0,0.4)+lf),
                                 int(hf*sigmoid(observation[0],2.5,0.6)+lf)]
        input_frequencies.extend(neurons_pole_velocity)

#        print("observation", observation)
#        print("input_frequencies", input_frequencies)

        return input_frequencies


    @staticmethod
    def _get_input_voltages(time_step: int, frequencies: List[int]):
        return [LIF_SPIKE_VOLTAGE if time_step > frequency and time_step % frequency == 0 else 0 for frequency in
                frequencies]

    @staticmethod
    def _count_spikes_within_time_window(time_step: int, actuator: List[int]):
        return len([t for t in actuator if time_step - t <= ACTUATOR_WINDOW])

    @staticmethod
    def _generate_spike_frequency(frequency: int) -> int:
        return int(1 / (TIME_STEP_IN_MSEC / 1000) / frequency)

