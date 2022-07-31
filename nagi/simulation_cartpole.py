#import random
#import re
from enum import Enum
from typing import List, Tuple

import numpy as np
import random

from nagi.constants_v2 import TIME_STEP_IN_MSEC, \
    ACTUATOR_WINDOW, LIF_SPIKE_VOLTAGE, MAX_POINTS_CARTPOLE_SIM, REPEAT_SIM
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
#        return [CartpoleConfigs.DEFAULT_M, CartpoleConfigs.INVERTED_M,
#                CartpoleConfigs.DEFAULT_XS, CartpoleConfigs.INVERTED_XS,
#                CartpoleConfigs.DEFAULT_XL, CartpoleConfigs.INVERTED_XL]
#        return [CartpoleConfigs.DEFAULT_M, CartpoleConfigs.DEFAULT_XS,
#                CartpoleConfigs.DEFAULT_XL, CartpoleConfigs.INVERTED_M,
#                CartpoleConfigs.INVERTED_XS, CartpoleConfigs.INVERTED_XL]
        return [CartpoleConfigs.DEFAULT_M,
                CartpoleConfigs.DEFAULT_XS,
                CartpoleConfigs.DEFAULT_XL]


    @staticmethod
    def get_testing_gates():
#        return [CartpoleConfigs.DEFAULT_S, CartpoleConfigs.INVERTED_S,
#                CartpoleConfigs.DEFAULT_L, CartpoleConfigs.INVERTED_L]
#        return [CartpoleConfigs.DEFAULT_S, CartpoleConfigs.DEFAULT_L,
#                CartpoleConfigs.INVERTED_S, CartpoleConfigs.INVERTED_L]
        return [CartpoleConfigs.DEFAULT_S,
                CartpoleConfigs.DEFAULT_L]

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

    def reset(self):
        self.spiking_neural_network.reset(reset_weights=True)

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
        self.current_env, self._mutator = self._initialize_logic_gate_and_mutator(testing)
        self.env_config_list = CartpoleConfigs.get_testing_gates() if testing else CartpoleConfigs.get_training_gates()
        self.maximum_possible_lifetime = MAX_POINTS_CARTPOLE_SIM * len(self.env_config_list)
        self.minimum_lifetime = 0

    def mutate(self):
        self.current_env = self._mutator[self.current_env]

    def simulate(self, agent: CartpoleAgent) -> Tuple[int, float]:
        reward_list = []
        env_list_idx = list(range(len(self.env_config_list)))
        random.shuffle(env_list_idx)
        print(self.env_config_list)
        print(self.env_config_list[0])
        print(self.env_config_list[-1])
        shuffled_env_list = [self.env_config_list[i] for i in env_list_idx]
        for i in range(REPEAT_SIM):
            agent.reset()
            #for env_id, env_config in enumerate(self.env_config_list):
            for env_id, env_config in enumerate(shuffled_env_list):
                env = NagiCartPoleEnv(lenght=env_config.value[0], inverted=env_config.value[1])
                observation = env.reset()
                agent.reset_actuators()
                left_actuator = []
                right_actuator = []
                current_reward = 0
                maximum_reward = 0

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
                    if reward == 1.0:
                        current_reward += reward
                    else:
                        current_reward = 0

                    if current_reward > maximum_reward:
                            maximum_reward = current_reward

                    if done:
                        break
                reward_list.append(maximum_reward / MAX_POINTS_CARTPOLE_SIM)
                env.close()
#            if env_id == int((len(self.env_config_list) / 2)-1) and np.mean(reward_list) < 0.5:
#                reward_list = np.sum(reward_list) / len(self.env_config_list)
#                break

        fitness = np.mean(reward_list)
        return (agent.key, fitness)

    def simulate_with_visualization(self, agent: CartpoleAgent) -> \
            Tuple[int, float, dict, dict, int, List[Tuple[int, int]], List[CartpoleConfigs], List[int]]:

        weights = {key: [] for key, _ in agent.spiking_neural_network.get_weights().items()}
        membrane_potentials = {key: [] for key, _ in
                               agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items()}

        actuator_logger = []
        env_config_logger = []
        reward = 0
        count_time_steps = 0
        flip_points_logger = []
        reward_list = []

        env_list_idx = list(range(len(self.env_config_list)))
        random.shuffle(env_list_idx)
        print(self.env_config_list)
        print(self.env_config_list[0])
        print(self.env_config_list[-1])
        shuffled_env_list = [self.env_config_list[i] for i in env_list_idx]
        # shuffled_env_list = [self.env_config_list[1], self.env_config_list[0]]
        for i in range(REPEAT_SIM):
            agent.reset()

            # for env_id, env_config in enumerate(self.env_config_list):
            for env_id, env_config in enumerate(shuffled_env_list):
                env = NagiCartPoleEnv(lenght=env_config.value[0], inverted=env_config.value[1])
                observation = env.reset()
                agent.reset_actuators()
                left_actuator = []
                right_actuator = []
                env_config_logger.append(env_config)

                current_reward = 0
                maximum_reward = 0
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
                    if reward == 1.0:
                        current_reward += reward
                    else:
                        current_reward = 0

                    if current_reward > maximum_reward:
                        maximum_reward = current_reward

                    print(f'Left: {agent.left_actuator}, Right: {agent.right_actuator}')
                    if done:
                        break
                reward_list.append(maximum_reward / MAX_POINTS_CARTPOLE_SIM)
                flip_points_logger.append(count_time_steps)
                #print( f'Pole length: {env_config.value[0]:.1f}, Inverted action: {env_config.value[1]}, Reward: {int(reward_list[-1])}')
                print( f'Pole length: {env_config.value[0]:.1f}, Inverted action: {env_config.value[1]}, Reward: {int(maximum_reward)}')
                env.close()
#            if env_id == int((len(self.env_config_list) / 2)-1) and np.mean(reward_list) < 0.5:
#                reward_list = np.sum(reward_list) / len(self.env_config_list)
#                break

        actuator_logger.append((agent.left_actuator, agent.right_actuator))
        for key, weight in agent.spiking_neural_network.get_weights().items():
            weights[key].append(weight)
        for key, membrane_potential in agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items():
            membrane_potentials[key].append(membrane_potential)

        fitness = np.mean(reward_list)
        print("Fitness:", fitness)
        print("Avg. steps:", fitness*MAX_POINTS_CARTPOLE_SIM)

        return (agent.key,
                fitness,
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
        # self.low_frequency = 2000
        # self.high_frequency = 200
        lf = self.low_frequency # Max 2000
        hf = self.high_frequency - self.low_frequency # Min 200

#        print("lf", lf)
#        print("hf", hf)

        neurons_cart_position = [int(hf*sigmoid(observation[0],-2.5,-0.6)+lf),
                                 int(hf*gaussian(observation[0],0,0.4)+lf),
                                 int(hf*sigmoid(observation[0],2.5,0.6)+lf)]
        input_frequencies.extend(neurons_cart_position)

        neurons_cart_velocity = [int(hf*sigmoid(observation[1],-2.5,-0.6)+lf),
                                 int(hf*gaussian(observation[1],0,0.4)+lf),
                                 int(hf*sigmoid(observation[1],2.5,0.6)+lf)]
        input_frequencies.extend(neurons_cart_velocity)

        neurons_pole_angle = [int(hf*sigmoid(observation[2],-60,-0.05)+lf),
                              int(hf*gaussian(observation[2],0,0.05)+lf),
                              int(hf*sigmoid(observation[2],60,0.05)+lf)]
        input_frequencies.extend(neurons_pole_angle)

        neurons_pole_velocity = [int(hf*sigmoid(observation[3],-2.5,-0.6)+lf),
                                 int(hf*gaussian(observation[3],0,0.4)+lf),
                                 int(hf*sigmoid(observation[3],2.5,0.6)+lf)]
        input_frequencies.extend(neurons_pole_velocity)

#        if time_step % ACTUATOR_WINDOW == 0:
#            print(time_step)
#            print("observation", observation)
#            print("input_frequencies", input_frequencies)

        # print("input_frequencies", input_frequencies)
        return input_frequencies

    @staticmethod
    def _initialize_logic_gate_and_mutator(testing: bool):
        if testing:
            gates = CartpoleConfigs.get_testing_gates()
        else:
            gates = CartpoleConfigs.get_training_gates()
        ordered_gates = random.sample(gates, len(gates))
        mutator = {}
        for i in range(0, len(ordered_gates) - 1):
            mutator[ordered_gates[i]] = ordered_gates[i + 1]
        mutator[ordered_gates[-1]] = ordered_gates[0]

        return ordered_gates[0], mutator

    @staticmethod
    def _get_input_voltages(time_step: int, frequencies: List[int]):
        return [LIF_SPIKE_VOLTAGE if time_step > frequency and time_step % frequency == 0 else 0 for frequency in
                frequencies]

    @staticmethod
    def _count_spikes_within_time_window(time_step: int, actuator: List[int]):
        # actuators_arr = np.asarray(actuator)
        # print("actuators_arr", np.mean(actuators_arr, axis=0))
        # print(actuators_arr.shape, time_step)
        return len([t for t in actuator if time_step - t <= ACTUATOR_WINDOW])

    @staticmethod
    def _generate_spike_frequency(frequency: int) -> int:
        return int(1 / (TIME_STEP_IN_MSEC / 1000) / frequency)

