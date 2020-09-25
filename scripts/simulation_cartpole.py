import pickle

import matplotlib.pyplot as plt
from easygui import fileopenbox
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import List

from definitions import ROOT_PATH
from nagi.constants import NUM_TIME_STEPS, RED, BLUE, GREEN
from nagi.simulation_cartpole import CartpoleEnvironment, CartpoleAgent
from nagi.visualization import visualize_genome

with open(f'{fileopenbox(default=f"{ROOT_PATH}/data/*genome*.pkl")}', 'rb') as file:
    test_genome = pickle.load(file)

agent = CartpoleAgent.create_agent(test_genome)
visualize_genome(test_genome, show_learning_rules=False, with_legend=False)
environment = CartpoleEnvironment(50, 5, testing=True)
(_,
 fitness,
 weights,
 membrane_potentials,
 time_step,
 actuators,
 env_config,
 flip_points_list) = environment.simulate_with_visualization(agent)

number_of_neurons = len(membrane_potentials.keys())
number_of_weights = len(weights.keys())
t_values = range(time_step + 1)
alpha = 0.85

def add_vertical_lines_and_background(height: int, flip_points: List[int]):
    for flip_point in flip_points:
        plt.axvline(x=flip_point, color='k')

def add_fig_legend(*args):
    colors = [arg[0] for arg in args]
    labels = [arg[1] for arg in args]
    plt.figlegend((*(Line2D([0], [0], color=color) for color in colors),
                   Line2D([0], [0], color='gray', linestyle='--'),
                   Line2D([0], [0], color='k'),
                   Patch(color=RED)),
                  (*(label for label in labels), 'sample change', 'flip point', 'wrong region'),
                  loc='upper left')


def add_table():
    the_table = plt.table(cellText=[[conf.value[0] for conf in env_config],
                                    [conf.value[1] for conf in env_config]],
                          rowLabels=['Pole length', 'Inverted'],
                          cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)


# Membrane potential
fig = plt.figure()
fig.suptitle("Neuron membrane potentials")
for i, key in enumerate(sorted(membrane_potentials.keys())):
    plt.subplot(number_of_neurons, 1, i + 1)
    h = plt.ylabel(f"({key})    ")
    h.set_rotation(0)
    if i == 0:
        plt.xlabel("Time step")
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
    else:
        plt.gca().axes.xaxis.set_visible(False)
    plt.plot(t_values, [membrane_potential[0] for membrane_potential in membrane_potentials[key]],
             color=GREEN, linestyle='-', alpha=alpha)
    plt.plot(t_values, [membrane_potential[1] for membrane_potential in membrane_potentials[key]],
             color=BLUE, linestyle='-', alpha=alpha)
    plt.xlim(0, len(t_values) + NUM_TIME_STEPS - len(t_values) % NUM_TIME_STEPS)
    add_fig_legend((GREEN, 'membrane potential'), (BLUE, 'membrane threshold'))
    add_vertical_lines_and_background(4, flip_points_list)
add_table()

# Weights
fig = plt.figure()
plt.suptitle("Weights")
for i, key in enumerate(sorted(weights.keys(), key=lambda x: x[1])):
    plt.subplot(number_of_weights, 1, i + 1)
    h = plt.ylabel(f"{key}       ")
    h.set_rotation(0)
    plt.gca().tick_params(axis='y', which='both', length=0, labelsize=0, labelcolor='#00000000')
    plt.ylim(-0.02, 1.02)

    if i == 0:
        plt.xlabel("Time step")
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
    else:
        plt.gca().axes.xaxis.set_visible(False)
    plt.plot(t_values, weights[key], color=BLUE, linestyle='-')
    add_fig_legend((BLUE, 'weight'))
    add_vertical_lines_and_background(2, flip_points_list)
    plt.xlim(0, len(t_values) + NUM_TIME_STEPS - len(t_values) % NUM_TIME_STEPS)
add_table()

# Actuator history
fig = plt.figure()
fig.suptitle("Actuator history")
left_actuators = [actuator[0] for actuator in actuators]
right_actuators = [actuator[1] for actuator in actuators]
plt.plot(t_values, left_actuators, color=GREEN, alpha=alpha)
plt.plot(t_values, right_actuators, color=BLUE, alpha=alpha)
plt.xlabel("Time step")
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
add_fig_legend((GREEN, 'left actuator'), (BLUE, 'right actuator'))
add_vertical_lines_and_background(max(max(left_actuators), max(right_actuators)) + 2, flip_points_list)
add_table()
plt.xlim(0, len(t_values) + NUM_TIME_STEPS - len(t_values) % NUM_TIME_STEPS)

print(f'\n **** Results ****')
print(f'Fitness: {fitness:.3f}')
plt.show()
