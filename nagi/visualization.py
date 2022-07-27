from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.text import Text
import os

from nagi.constants import RED, BLUE, GREEN, PINK, CYAN
from nagi.neat import Genome, NeuralNodeGene, NodeGene, InputNodeGene, \
    OutputNodeGene, HiddenNodeGene, NeuralNodeGeneDoublePlasticity, \
    OutputNodeGeneDoublePlasticity, HiddenNodeGeneDoublePlasticity, \
    GenomeDoublePlasticity
import time

timestr = time.strftime("%Y%m%d-%H%M%S")


class CustomTextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        tx = Text(width/2, height/2, orig_handle.get_text(), fontsize=fontsize,
                  ha="center", va="center", fontweight="normal")
        return [tx]


def visualize_genome(genome: Genome, show_learning_rules: bool = True, with_legend: bool = True):
    def get_color(node: NodeGene):
        if isinstance(node, InputNodeGene):
            return GREEN
        elif isinstance(node, (HiddenNodeGene, OutputNodeGene)):
            if node.is_inhibitory:
                if node.bias:
                    return PINK
                else:
                    return RED
            else:
                if node.bias:
                    return CYAN
                else:
                    return BLUE

    def legend_circle(color: str):
        return Line2D([0], [0], color=color, marker='o', linewidth=0)
    plt.figure(figsize=(8,8))
    g, nodes, edges = genome_to_graph(genome)
    pos = get_node_coordinates(genome)

    labels = {
        #key: f"{node.learning_rule.value if isinstance(node, NeuralNodeGene) else key}{'↩' if (key, key) in edges else ''}"
        key: f"{node.learning_rule.value if isinstance(node, NeuralNodeGene) else key}{'' if (key, key) in edges else ''}"
        for key, node in genome.nodes.items()} if show_learning_rules \
        else {node: f"{node}{'' if (node, node) in edges else ''}" for node in nodes}
        #else {node: f"{node}{'↩' if (node, node) in edges else ''}" for node in nodes}

    node_color = [get_color(genome.nodes[node]) for node in nodes]
    edgecolors = ['k' if isinstance(genome.nodes[node], OutputNodeGene) else get_color(genome.nodes[node]) for node in nodes]
    #nx.draw_networkx_nodes(g, pos=pos, nodes=nodes, node_color=node_color, edgecolors=edgecolors, node_size=400)
    nx.draw_networkx_nodes(g, pos=pos, node_color=node_color, edgecolors=edgecolors, node_size=400)
    nx.draw_networkx_labels(g, pos=pos, labels=labels)
    nx.draw_networkx_edges(g, pos=pos, connectionstyle="arc3, rad=0.05")
    # nx.draw_networkx(g, pos=pos, with_labels=True, labels=labels, nodes=nodes, node_color=node_color, node_size=400,
    #                  font_size=10, connectionstyle="arc3, rad=0.05")

    Legend.update_default_handler_map({Text: CustomTextHandler()})
    legend_dict = {legend_circle(GREEN): 'input node',
                   Line2D([0], [0], color='w', markeredgecolor='k', marker='o', linewidth=0): 'output node',
                   legend_circle(BLUE): 'excitatory, without bias',
                   legend_circle(RED): 'inhibitory, without bias',
                   legend_circle(CYAN): 'excitatory, with bias',
                   legend_circle(PINK): 'inhibitory, with bias',
                   Text(text='AH'): 'asymmetric hebbian',
                   Text(text='AA'): 'asymmetric anti-hebbian',
                   Text(text='SH'): 'symmetric hebbian',
                   Text(text='SA'): 'symmetric anti-hebbian'}
    if with_legend:
        plt.figlegend(handles=legend_dict.keys(), labels=legend_dict.values(), loc='upper right')
    plt.box(False)
    plt.show()

def visualize_genome_doubleplast(genome: GenomeDoublePlasticity,
                                 show_learning_rules: bool = True,
                                 with_legend: bool = True,
                                 savefig: bool = False):
    def get_color(node: NodeGene):
        if isinstance(node, InputNodeGene):
            return GREEN
        elif isinstance(node, (HiddenNodeGeneDoublePlasticity, OutputNodeGeneDoublePlasticity)):
            if node.is_inhibitory:
                if node.bias:
                    return PINK
                else:
                    return RED
            else:
                if node.bias:
                    return CYAN
                else:
                    return BLUE

    def legend_circle(color: str):
        return Line2D([0], [0], color=color, marker='o', linewidth=0)

    g, nodes, edges = genome_to_graph(genome)
    pos = get_node_coordinates(genome)

    labels = {
        key: f"{node.learning_rule.value+'i'+node.learning_rule_inh.value if isinstance(node, NeuralNodeGene) else key}{'↩' if (key, key) in edges else ''}"
        for key, node in genome.nodes.items()} if show_learning_rules \
        else {node: f"{node}{'↩' if (node, node) in edges else ''}" for node in nodes}

    node_color = [get_color(genome.nodes[node]) for node in nodes]
    edgecolors = ['k' if isinstance(genome.nodes[node], OutputNodeGeneDoublePlasticity) else get_color(genome.nodes[node]) for node in nodes]
    #nx.draw_networkx_nodes(g, pos=pos, nodes=nodes, node_color=node_color, edgecolors=edgecolors, node_size=400)
    nx.draw_networkx_nodes(g, pos=pos, node_color=node_color, edgecolors=edgecolors, node_size=400)
    nx.draw_networkx_labels(g, pos=pos, labels=labels)
    nx.draw_networkx_edges(g, pos=pos, connectionstyle="arc3, rad=0.05")
    # nx.draw_networkx(g, pos=pos, with_labels=True, labels=labels, nodes=nodes, node_color=node_color, node_size=400,
    #                  font_size=10, connectionstyle="arc3, rad=0.05")

    Legend.update_default_handler_map({Text: CustomTextHandler()})
    legend_dict = {legend_circle(GREEN): 'input node',
                   Line2D([0], [0], color='w', markeredgecolor='k', marker='o', linewidth=0): 'output node',
                   legend_circle(BLUE): 'excitatory, without bias',
                   legend_circle(RED): 'inhibitory, without bias',
                   legend_circle(CYAN): 'excitatory, with bias',
                   legend_circle(PINK): 'inhibitory, with bias',
                   Text(text='AH'): 'asymmetric hebbian',
                   Text(text='AA'): 'asymmetric anti-hebbian',
                   Text(text='SH'): 'symmetric hebbian',
                   Text(text='SA'): 'symmetric anti-hebbian'}
    if with_legend:
        plt.figlegend(handles=legend_dict.keys(), labels=legend_dict.values(), loc='upper right')
    plt.box(False)
    if savefig:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/topology_4d_doubleplast_'+timestr+'.png')   # save the figure to file
    else:
        plt.show()


def genome_to_graph(genome: Genome):
    edges = [(connection.origin_node, connection.destination_node)
             for connection in genome.get_enabled_connections()]
    nodes = [key for key in genome.nodes.keys()]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g, nodes, edges


def get_node_coordinates(genome: Genome):
    def layer_y_linspace(start, end):
        if number_of_nodes == 1:
            return np.mean((start, end))
        else:
            return np.linspace(start, end, number_of_nodes)

    def sort_by_layers():
        keys_with_layers = list(zip(sorted(genome.nodes.keys()), layers))
        return [key for key, _ in sorted(keys_with_layers, key=lambda tup: tup[1])]

    figure_width = 10
    figure_height = 5
    layers = get_layers(genome)
    x = layers / max(layers) * figure_width
    _, number_of_nodes_per_layer = np.unique(layers, return_counts=True)
    y = np.array([])
    for number_of_nodes in number_of_nodes_per_layer:
        margin = figure_height / (number_of_nodes ** 1.5)
        y = np.r_[y, layer_y_linspace(margin, figure_height - margin)]

    y_coords = {key: y for key, y in zip(sort_by_layers(), y)}
    return {key: (x_coord, y_coords[key]) for key, x_coord in zip(sorted(genome.nodes.keys()), x)}


def get_layers(genome: Genome):
    """
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1
    """
    adjacency_matrix = get_adjacency_matrix(genome)
    adjacency_matrix[:, genome.input_size: genome.input_size + genome.output_size] = 0
    n_node = np.shape(adjacency_matrix)[0]
    layers = np.zeros(n_node)
    while True:  # Loop until sorting doesn't help any more
        prev_order = np.copy(layers)
        for curr in range(n_node):
            src_layer = np.zeros(n_node)
            for src in range(n_node):
                src_layer[src] = layers[src] * adjacency_matrix[src, curr]
            layers[curr] = np.max(src_layer) + 1
        if all(prev_order == layers):
            break
    set_final_layers(layers, genome.input_size, genome.output_size)
    return layers


def get_adjacency_matrix(genome: Genome):
    n = len(genome.nodes)
    node_order_map = {key: i for i, key in enumerate(sorted(genome.nodes.keys()))}
    adjacency_matrix = np.zeros((n, n))
    genome_copy = deepcopy(genome)
    connections_to_ignore = get_last_connection_in_all_cycles(genome_copy)

    for connection in connections_to_ignore:
        genome_copy.connections.pop(connection)
    for connection in genome_copy.get_enabled_connections():
        adjacency_matrix[node_order_map[connection.origin_node]][node_order_map[connection.destination_node]] = 1
    return adjacency_matrix


def get_last_connection_in_all_cycles(genome: Genome):
    return set([max(cycle) for cycle in get_simple_cycles(genome)])


def get_simple_cycles(genome: Genome):
    def cycle_to_list_of_tuples(cycle):
        cycle.append(cycle[0])
        return [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]

    edge_to_innovation_number_map = {(connection.origin_node, connection.destination_node): connection.innovation_number
                                     for connection in genome.connections.values()}
    simple_cycles = [cycle_to_list_of_tuples(cycle) for cycle in nx.simple_cycles(genome_to_graph(genome)[0])]
    return [[edge_to_innovation_number_map[edge] for edge in cycle] for cycle in simple_cycles]


def set_final_layers(layers: np.ndarray, input_size: int, output_size: int):
    max_layer = max(layers) + 1
    for i in range(len(layers)):
        if i < input_size:
            layers[i] = 1
        elif i < input_size + output_size:
            layers[i] = max_layer
        elif layers[i] == 1:
            layers[i] = 2
