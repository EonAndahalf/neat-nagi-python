from enum import Enum
from typing import List, Dict

import numpy as np

from nagi.constants import STDP_PARAMS, STDP_LEARNING_WINDOW, NEURON_WEIGHT_BUDGET, \
    THRESHOLD_THETA_INCREMENT_RATE, THRESHOLD_THETA_DECAY_RATE, \
    LIF_RESTING_MEMBRANE_POTENTIAL, LIF_MEMBRANE_POTENTIAL_THRESHOLD, LIF_SPIKE_VOLTAGE, \
    LIF_MEMBRANE_DECAY_RATE, LIF_BIAS, WEIGHT_INIT_MEAN, WEIGHT_INIT_STD
from nagi.neat import Genome, NeuralNodeGene, InputNodeGene, OutputNodeGene, \
    GenomeDoublePlasticity, NeuralNodeGeneDoublePlasticity, \
    OutputNodeGeneDoublePlasticity
from nagi.stdp import *


class StdpType(Enum):
    input = 1
    output = 2


class LIFSpikingNeuron(object):
    """Class representing a single spiking neuron."""

    def __init__(self, inputs: List[int], learning_rule: LearningRule, is_inhibitory: bool,
                 stdp_parameters: Dict[str, float], bias: bool):
        """
        :param inputs: A dictionary of incoming connection weights.
        """

        self.inputs = {key: np.random.normal(WEIGHT_INIT_MEAN, WEIGHT_INIT_STD) for key in inputs}
        self._normalize_weights()
        self.learning_rule = learning_rule
        self.is_inhibitory = is_inhibitory
        self.stdp_parameters = stdp_parameters
        self.bias = LIF_BIAS if bias else 0

        self.membrane_potential = LIF_RESTING_MEMBRANE_POTENTIAL
        self.fired = 0
        self.threshold_theta = 0

        # Variables containing time elapsed since last input and output spikes.
        self.output_spike_timing: float = 0
        self.input_spike_timings: Dict[int, List[float]] = {key: [] for key in self.inputs.keys()}
        self.has_fired = False

    def advance(self, dt: float):
        """
        Advances simulation time by the given time step in milliseconds.

        :param dt: Time step in milliseconds.
        """

        if self.fired:
            self.membrane_potential = LIF_RESTING_MEMBRANE_POTENTIAL
            self.threshold_theta += THRESHOLD_THETA_INCREMENT_RATE

        self.fired = 0
        self.output_spike_timing += dt

        for key in self.input_spike_timings.keys():
            # STDP update on received input spike.
            if 0 in self.input_spike_timings[key] and self.has_fired:
                self.stpd_update(key, StdpType.input)

            self.input_spike_timings[key] = [t + dt for t in self.input_spike_timings[key] if
                                             t + dt < STDP_LEARNING_WINDOW]

        if self.membrane_potential > self.get_threshold() + self.threshold_theta:
            self.fired = LIF_SPIKE_VOLTAGE if not self.is_inhibitory else -LIF_SPIKE_VOLTAGE
            self.has_fired = True
            self.output_spike_timing = 0

            # STDP on output spike.
            for key in self.input_spike_timings.keys():
                self.stpd_update(key, StdpType.output)
        else:
            self.threshold_theta -= THRESHOLD_THETA_DECAY_RATE * self.threshold_theta

    def reset(self, reset_weights=False):
        """ Resets all state variables."""

        self.membrane_potential = LIF_RESTING_MEMBRANE_POTENTIAL
        self.fired = 0
        self.output_spike_timing = 0
        self.input_spike_timings = {key: [] for key in self.inputs.keys()}
        if reset_weights:
            self.inputs = {key: np.random.normal(WEIGHT_INIT_MEAN, WEIGHT_INIT_STD) for key in self.inputs}

    def apply_learning_rule(self, delta_t: float):
        return get_learning_rule_function(self.learning_rule)(delta_t, **self.stdp_parameters)

    def stpd_update(self, key: int, stdp_type: StdpType):
        """
        Applies STDP to the weight with the supplied key.

        :param stdp_type:
        :param key: The key identifying the synapse weight to be updated.
        :return: void
        """

        delta_weight = 0
        weight = self.inputs[key]
        sigma, w_min, w_max = STDP_PARAMS['sigma'], STDP_PARAMS['w_min'], STDP_PARAMS['w_max']

        if stdp_type is StdpType.input:
            delta_t = self.output_spike_timing - 0
            if abs(delta_t) < STDP_LEARNING_WINDOW:
                delta_weight = self.apply_learning_rule(delta_t)

        elif stdp_type is StdpType.output:
            for input_spike_timing in self.input_spike_timings[key]:
                delta_t = self.output_spike_timing - input_spike_timing
                if abs(delta_t) < STDP_LEARNING_WINDOW:
                    delta_weight += self.apply_learning_rule(delta_t)

        if delta_weight > 0:
            self.inputs[key] += sigma * delta_weight * (w_max - weight)
        elif delta_weight < 0:
            self.inputs[key] += sigma * delta_weight * (weight - abs(w_min))

        self._normalize_weights()

    def _normalize_weights(self):
        sum_of_input_weights = sum(self.inputs.values())
        if sum_of_input_weights > NEURON_WEIGHT_BUDGET:
            self.inputs = {key: value * NEURON_WEIGHT_BUDGET / sum_of_input_weights for key, value in
                           self.inputs.items()}

    def get_threshold(self):
        return min(sum(self.inputs.values()), LIF_MEMBRANE_POTENTIAL_THRESHOLD)

class LIFSpikingNeuronDoublePlasticity(LIFSpikingNeuron):
    def __init__(self, inputs: List[int], inputs_inh: List[bool],
                 learning_rule_exc: LearningRule,
                 learning_rule_inh: LearningRule, is_inhibitory: bool,
                 stdp_parameters_exc: Dict[str, float],
                 stdp_parameters_inh: Dict[str, float],
                 bias: bool):
        super().__init__(inputs, learning_rule_exc, is_inhibitory,
                 stdp_parameters_exc, bias)
        self.learning_rule_inh = learning_rule_inh
        self.stdp_parameters_inh = stdp_parameters_inh
        self.inputs_inh = {key: inh for key,inh in zip(inputs,inputs_inh)}

    def apply_learning_rule(self, delta_t: float):
        return get_learning_rule_function(self.learning_rule)(delta_t, **self.stdp_parameters),\
                  get_learning_rule_function(self.learning_rule_inh)(delta_t, **self.stdp_parameters_inh)

    def stpd_update(self, key: int, stdp_type: StdpType):
        """
        Applies STDP to the weight with the supplied key.

        :param stdp_type:
        :param key: The key identifying the synapse weight to be updated.
        :return: void
        """

        delta_weight = 0
        weight = self.inputs[key]
        sigma, w_min, w_max = STDP_PARAMS['sigma'], STDP_PARAMS['w_min'], STDP_PARAMS['w_max']

        if stdp_type is StdpType.input:
            delta_t = self.output_spike_timing - 0
            if abs(delta_t) < STDP_LEARNING_WINDOW:
                if not self.inputs_inh[key]:
                    delta_weight = self.apply_learning_rule(delta_t)[0]
                else:
                    delta_weight = self.apply_learning_rule(delta_t)[1]

        elif stdp_type is StdpType.output:
            for input_spike_timing in self.input_spike_timings[key]:
                delta_t = self.output_spike_timing - input_spike_timing
                if abs(delta_t) < STDP_LEARNING_WINDOW:
                    if not self.inputs_inh[key]:
                        delta_weight += self.apply_learning_rule(delta_t)[0]
                    else:
                        delta_weight += self.apply_learning_rule(delta_t)[1]

        if delta_weight > 0:
            self.inputs[key] += sigma * delta_weight * (w_max - weight)
        elif delta_weight < 0:
            self.inputs[key] += sigma * delta_weight * (weight - abs(w_min))

        self._normalize_weights()

class LIFSpikingNeuralNetwork(object):
    """Class representing a spiking neural network."""

    def __init__(self, neurons: Dict[int, LIFSpikingNeuron], inputs: List[int], outputs: List[int]):
        """
        :param neurons: Dictionary containing key/node pairs.
        :param inputs: List of input node keys.
        :param outputs: List of output node keys.
        :var self.input_values: Dictionary containing input key/voltage pairs.
        """

        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.input_values: Dict[int, float] = {}
        self.number_of_hidden_neurons = len(self.neurons) - len(outputs)

    def set_inputs(self, inputs: List[float]):
        """
        Assigns voltages to the input nodes.

        :param inputs: List of voltage values."""

        assert len(inputs) == len(
            self.inputs), f"Number of inputs {len(inputs)} does not match number of input nodes {len(self.inputs)} "

        for key, voltage in zip(self.inputs, inputs):
            self.input_values[key] = voltage

    def advance(self, dt: float) -> List[float]:
        """
        Advances the neural network with the given input values and neuron states. Iterates through each neuron, then
        through each input of each neuron and evaluates the values to advance the network. The values can come from
        either input nodes, or firing neurons in a previous layer.

        :param dt: Time step in miliseconds.
        :return: List of the output values of the network after advance."""

        for neuron in self.neurons.values():
            sum_of_inputs = 0
            for key, weight in neuron.inputs.items():
                in_neuron = self.neurons.get(key)
                if in_neuron is not None:
                    in_value = in_neuron.fired
                else:
                    in_value = self.input_values[key]

                # Trigger STDP on received input spike.
                if in_value:
                    neuron.input_spike_timings[key].append(0)

                sum_of_inputs += weight * in_value + neuron.bias

            neuron.membrane_potential = max(
                neuron.membrane_potential + (sum_of_inputs - neuron.membrane_potential * LIF_MEMBRANE_DECAY_RATE),
                LIF_RESTING_MEMBRANE_POTENTIAL)

        for neuron in self.neurons.values():
            neuron.advance(dt)

        return [self.neurons[key].fired for key in self.outputs]

    def reset(self, reset_weights=False):
        """Resets all state variables in all neurons in the entire neural network."""
        for neuron in self.neurons.values():
            neuron.reset(reset_weights=reset_weights)

    def get_weights(self):
        weights = {}
        for destination_key, neuron in self.neurons.items():
            for origin_key, weight in neuron.inputs.items():
                weights[(origin_key, destination_key)] = weight
        return weights

    def get_membrane_potentials_and_thresholds(self):
        return {key: (neuron.membrane_potential, neuron.get_threshold() + neuron.threshold_theta) for key, neuron
                in self.neurons.items()}

    @staticmethod
    def create(genome: Genome):
        learning_nodes = {key: node for key, node in genome.nodes.items() if isinstance(node, NeuralNodeGene)}
        node_inputs = {key: [] for key in learning_nodes.keys()}
        input_keys = [node.key for node in genome.nodes.values() if isinstance(node, InputNodeGene)]
        output_keys = [node.key for node in genome.nodes.values() if isinstance(node, OutputNodeGene)]

        for connection_gene in genome.get_enabled_connections():
            node_inputs[connection_gene.destination_node].append(connection_gene.origin_node)

        # try block needed for legacy genomes where nodes lack bias.
        # TODO: Remove except at later point.
        try:
            neurons = {key: LIFSpikingNeuron(inputs,
                                             learning_nodes[key].learning_rule,
                                             learning_nodes[key].is_inhibitory,
                                             learning_nodes[key].stdp_parameters,
                                             learning_nodes[key].bias)
                       for key, inputs in node_inputs.items()}

        except AttributeError:
            neurons = {key: LIFSpikingNeuron(inputs,
                                             learning_nodes[key].learning_rule,
                                             learning_nodes[key].is_inhibitory,
                                             learning_nodes[key].stdp_parameters,
                                             False)
                       for key, inputs in node_inputs.items()}

        return LIFSpikingNeuralNetwork(neurons, input_keys, output_keys)

class LIFSpikingNeuralNetworkDoublePlasticity(LIFSpikingNeuralNetwork):
    """Class representing a spiking neural network
    with double plasticity neurons."""

    def __init__(self, neurons: Dict[int, LIFSpikingNeuronDoublePlasticity],
                 inputs: List[int], outputs: List[int]):
        """
        :param neurons: Dictionary containing key/node pairs.
        :param inputs: List of input node keys.
        :param outputs: List of output node keys.
        :var self.input_values: Dictionary containing input key/voltage pairs.
        """
        super().__init__(neurons, inputs, outputs)


    @staticmethod
    def create(genome: GenomeDoublePlasticity):
        learning_nodes = {key: node for key, node in genome.nodes.items() if isinstance(node, NeuralNodeGeneDoublePlasticity)}
        node_inputs = {key: [] for key in learning_nodes.keys()}
        node_inputs_inh = {key: [] for key in learning_nodes.keys()}
        input_keys = [node.key for node in genome.nodes.values() if isinstance(node, InputNodeGene)]
        output_keys = [node.key for node in genome.nodes.values() if isinstance(node, OutputNodeGeneDoublePlasticity)]
        #print(genome.get_enabled_connections())
        for connection_gene in genome.get_enabled_connections():
            #print(connection_gene.origin_node, connection_gene.destination_node)
            node_inputs[connection_gene.destination_node].append(connection_gene.origin_node)

            if connection_gene.origin_node in learning_nodes:
                node_inputs_inh[connection_gene.destination_node].append(
                    learning_nodes[connection_gene.origin_node].is_inhibitory)
            else: # Input Node is always excitatory
                node_inputs_inh[connection_gene.destination_node].append(False)

        # try block needed for legacy genomes where nodes lack bias.
        # TODO: Remove except at later point.
        try:
            neurons = {key: LIFSpikingNeuronDoublePlasticity(inputs,
                                             node_inputs_inh[key],
                                             learning_nodes[key].learning_rule,
                                             learning_nodes[key].learning_rule_inh,
                                             learning_nodes[key].is_inhibitory,
                                             learning_nodes[key].stdp_parameters,
                                             learning_nodes[key].stdp_parameters_inh,
                                             learning_nodes[key].bias)
                       for key, inputs in node_inputs.items()}

        except AttributeError:
            neurons = {key: LIFSpikingNeuronDoublePlasticity(inputs,
                                             node_inputs_inh[key],
                                             learning_nodes[key].learning_rule,
                                             learning_nodes[key].learning_rule_inh,
                                             learning_nodes[key].is_inhibitory,
                                             learning_nodes[key].stdp_parameters,
                                             learning_nodes[key].stdp_parameters_inh,
                                             False)
                       for key, inputs in node_inputs.items()}

        return LIFSpikingNeuralNetworkDoublePlasticity(neurons, input_keys, output_keys)