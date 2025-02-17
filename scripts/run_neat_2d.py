import multiprocessing as mp
import time, pickle, os, sys, tqdm
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Fucking windows 98 cant find its own paths
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


from nagi.neat import Population
from nagi.simulation_2d import TwoDimensionalAgent, TwoDimensionalEnvironment


def get_file_paths():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return f'{ROOT_PATH}/data/2D_test_run_{timestr}.pkl', f'{ROOT_PATH}/data/2D_test_run_{timestr}_config.txt'


def generate_config_string():
    with open(f'{ROOT_PATH}/nagi/constants.py', 'r') as f:
        constants = f.read()

    return f"""Environment type: {environment_type}
Input size: {input_size}
Output size: {output_size}
Population size: {population_size}
Number of generations: {number_of_generations}
High frequency: {high_frequency}
Low frequency: {low_frequency}

{constants}"""


environment_type = "2D"
input_size, output_size = 6, 2
high_frequency = 50
low_frequency = 5

population_size = 100
number_of_generations = 500

if __name__ == '__main__':
    pickle_path, txt_path = get_file_paths()

    with open(txt_path, 'w') as file:
        file.write(generate_config_string())

    pool = mp.Pool(mp.cpu_count())

    population = Population(population_size, input_size, output_size)
    generations = {}
    for i in range(0, number_of_generations):
        print(f'\nGeneration {i}...')

        env = TwoDimensionalEnvironment(high_frequency, low_frequency)
        test_env = TwoDimensionalEnvironment(high_frequency, low_frequency, testing=True)
        agents = list([TwoDimensionalAgent.create_agent(genome) for genome in population.genomes.values()])

        results = tqdm.tqdm(pool.imap_unordered(env.simulate, agents), total=(len(agents)))
        data_dict = {result[0]: result[1:] for result in results}
        
        fitnesses = {key: value[0] for key, value in data_dict.items()}
        accuracies = {key: value[1] for key, value in data_dict.items()}

        end_of_sample_accuracies = {key: value[2] for key, value in data_dict.items()}

        most_fit_genome_key, highest_fitness = max(fitnesses.items(), key=lambda x: x[1])
        most_acc_genome_key, highest_accuracy = max(accuracies.items(), key=lambda x: x[1])
        most_eos_acc_genome_key, highest_eos_accuracy = max(accuracies.items(), key=lambda x: x[1])

        print(f'Highest fitness: {highest_fitness:.3f}')
        print(f'Highest accuracy: {highest_accuracy * 100:.1f}%')
        print(f'Highest end-of-sample accuracy: {highest_eos_accuracy * 100:.1f}%')

        print('Running test simulations...')
        test_agents = [TwoDimensionalAgent.create_agent(population.genomes[key]) for key in
                       [most_fit_genome_key, most_acc_genome_key, most_eos_acc_genome_key]]
        test_results = list(pool.imap(test_env.simulate, test_agents))
        test_fitness = test_results[0][1]
        test_accuracy = test_results[1][2]
        test_eos_accuracy = test_results[2][3]

        print(f'Test fitness: {test_fitness:.3f}')
        print(f'Test accuracy: {test_accuracy * 100:.1f}%')
        print(f'Test end-of-sample accuracy: {test_eos_accuracy * 100:.1f}%')

        # TODO: Don't actually use the 0th element, works with older test data. Fix to a 3-tuple at a later point.
        test_result = (0, test_fitness, test_accuracy, test_eos_accuracy)

        generations[i] = {'population': deepcopy(population),
                          'fitnesses': deepcopy(fitnesses),
                          'accuracies': accuracies,
                          'end_of_sample_accuracies': end_of_sample_accuracies,
                          'test_result': test_result}

        with open(pickle_path, 'wb') as file:
            pickle.dump(generations, file)

        population.next_generation(fitnesses)
