import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
import pandas as pd


def hyperparam_test(seed):
    name = 'mimicparams'
    population_sizes = [100*n for n in range(1,11)] # 100 through 1000
    keep_percent_list = [0.1*n for n in range(1,6)] # 0.1 through 0.5
    
    lengths = [10*n for n in range(4,11)]
    iteration_list = [n for n in range(101)]

    for length in lengths:
        print(f'\nBeginning experiments with {length}-element vector.')

        problem = mlrose_hiive.MaxKColorGenerator.generate(seed=seed, number_of_nodes=length, max_colors=5)

        runner = mlrose_hiive.MIMICRunner(problem=problem, experiment_name='MIMIC', seed=seed, iteration_list=iteration_list, 
                                           population_sizes=population_sizes, keep_percent_list=keep_percent_list, 
                                           max_attempts=30, use_fast_mimic=True)
        
        run_stats, curves = runner.run()

        with pd.ExcelWriter(f'{runner._experiment_name}_{name}_{length}.xlsx') as writer:
            run_stats.to_excel(writer, sheet_name='Run Stats')
            curves.to_excel(writer, sheet_name='Curves')
        



    if False:
        # Iteration vs Fitness
        for key in fitness_development:
            plt.plot(range(len(fitness_development[key])), -1*np.array(fitness_development[key]), label=key, color=random.choice(colors))

        zero_line = [0 for n in range(len(fitness_development[max(fitness_development, key=lambda key: len(fitness_development[key]))]))]
        plt.plot(range(len(zero_line)), zero_line, label='Upper Bound', color='gold', linestyle=random.choice(linestyles))
        plt.legend(loc='best')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title(f'Iteration vs. Fitness for Max K Colors')
        plt.savefig('Graphs/kcolor/iteration_v_fitness_max_k_params.png')
        # plt.show()
        plt.close()


def run_and_plot(seed=1, generate_plots=True):
    lengths = [5*n for n in range(1,21)]

    iteration_list = [n for n in range(10001)] # iterations at which to log results.
    fixed_length = 35 # for this length, plot iteration versus fitness

    global_optima = [0 for n in range(len(lengths))]

    # SA parameters
    temperature_list = [1] # initial temperature -- higher means more exploration
    decay_list = [mlrose_hiive.ExpDecay] # slower decay means more exploration. exp decays slower than geom.

    # GA parameters
    population_sizes = [200]
    mutation_rates = [0.2]

    # RHC parameters
    restart_list = [20]

    # MIMIC parameters
    population_sizes = [400]
    keep_percent_list = [0.1]

    # log best fitnesses acheived for each algorithm, to plot later
    best_fitnesses = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}
    
    times = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}

    # log fevals for each algorithm, to plot later
    fevals = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}
    
    # fitness score development for set length
    fitness_development = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}

    for length in lengths:
        print(f'Starting length {length}')
        problem = mlrose_hiive.MaxKColorGenerator.generate(seed=seed, number_of_nodes=length, max_colors=5)
        
        runners = [mlrose_hiive.SARunner(problem=problem, experiment_name='SA', seed=seed, iteration_list=iteration_list, 
                                        temperature_list=temperature_list, decay_list=decay_list, max_attempts=10),

                    mlrose_hiive.GARunner(problem=problem, experiment_name='GA', seed=seed, iteration_list=iteration_list, 
                                            population_sizes=population_sizes, mutation_rates=mutation_rates, max_attempts=10),

                    mlrose_hiive.RHCRunner(problem=problem, experiment_name='RHC', seed=seed, iteration_list=iteration_list, 
                                            restart_list=restart_list, max_attempts=10),

                    mlrose_hiive.MIMICRunner(problem=problem, experiment_name='MIMIC', seed=seed, iteration_list=iteration_list, 
                                                population_sizes=population_sizes, keep_percent_list=keep_percent_list, 
                                                use_fast_mimic=True, max_attempts=10)]
        
        for runner in runners:
            run_stats, curves = runner.run()

            # compiling length versus best fitness achieved plot
            best_fitnesses[runner._experiment_name].append(min(curves['Fitness']))

            # compiling length versus clock time plot
            times[runner._experiment_name].append(curves['Time'].iloc[-1] - curves['Time'].iloc[0])

            # compiling length versus f-evals plot
            eval_count_RHC = sum(curves['FEvals'].iloc[find_peaks(curves['FEvals'])[0]])
            eval_count_others = curves['FEvals'].iloc[-1]
            eval_count = max(eval_count_RHC, eval_count_others)
            fevals[runner._experiment_name].append(eval_count)

            # plot iteration versus fitness for the set length
            if length == fixed_length:
                fitness_development[runner._experiment_name] = curves['Fitness']


    if generate_plots:
        # Iteration vs Fitness
        zero_line = [0 for n in range(max(len(fitness_development['SA']),len(fitness_development['GA']), len(fitness_development['RHC']), len(fitness_development['MIMIC'])))]
        plt.plot(range(len(fitness_development['RHC'])), -1*np.array(fitness_development['RHC']), label='RHC', color='mediumpurple')
        plt.plot(range(len(fitness_development['SA'])), -1*np.array(fitness_development['SA']), label='SA', color='cadetblue')
        plt.plot(range(len(fitness_development['GA'])), -1*np.array(fitness_development['GA']), label='GA', color='yellowgreen')
        plt.plot(range(len(fitness_development['MIMIC'])), -1*np.array(fitness_development['MIMIC']), label='MIMIC', color='lightpink')
        plt.plot(range(len(zero_line)), zero_line, label='Upper Bound', color='gold', linestyle='dashed')
        plt.legend(loc='best')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title(f'Iteration vs. Fitness for Max K Color of Length {fixed_length}')
        plt.savefig('iteration_v_fitness_k_color.png')
        # plt.show()
        plt.close()

        # Length vs Fitness
        plt.plot(lengths, global_optima[:len(lengths)], label='Upper Bound', color='gold', linewidth=3, alpha=0.5)
        plt.plot(lengths, -1*np.array(best_fitnesses['RHC']), label='RHC', color='mediumpurple', marker='*')
        plt.plot(lengths, -1*np.array(best_fitnesses['GA']), label='GA', color='yellowgreen', marker='o')
        plt.plot(lengths, -1*np.array(best_fitnesses['SA']), label='SA', color='cadetblue', marker=6)
        plt.plot(lengths, -1*np.array(best_fitnesses['MIMIC']), label='MIMIC', color='lightpink')
        plt.legend(loc='best')
        plt.xlabel('Problem Length (bits)')
        plt.ylabel('Fitness')
        plt.title('Length vs. Fitness on Max K Color')
        plt.savefig('length_v_fitness_k_color.png')
        # plt.show()
        plt.close()

        # Length vs F-Evals
        plt.plot(lengths, fevals['RHC'], label='RHC', color='mediumpurple')
        plt.plot(lengths, fevals['SA'], label='SA', color='cadetblue')
        plt.plot(lengths, fevals['GA'], label='GA', color='yellowgreen')
        plt.plot(lengths, fevals['MIMIC'], label='MIMIC', color='lightpink')
        plt.legend(loc='best')
        plt.xlabel('Problem Length (bits)')
        plt.ylabel('Fitness Function Evaluations')
        plt.title('Length vs. Fitness Function Evaluations on Max K Color')
        plt.savefig('length_v_fevals_k_color.png')
        # plt.show()
        plt.close()

        # Length vs Time
        plt.plot(lengths, times['RHC'], label='RHC', color='mediumpurple')
        plt.plot(lengths, times['SA'], label='SA', color='cadetblue')
        plt.plot(lengths, times['GA'], label='GA', color='yellowgreen')
        plt.plot(lengths, times['MIMIC'], label='MIMIC', color='lightpink')
        plt.legend(loc='best')
        plt.xlabel('Problem Length (bits)')
        plt.ylabel('Time Taken (seconds)')
        plt.title('Length vs. Time on Max K Color')
        plt.savefig('length_v_time_k_color.png')
        # plt.show()
        plt.close()

        print('Done.')
    
    return fitness_development, best_fitnesses, fevals


def generate_average_fitness_plot():
    np.random.seed(1)
    num_runs = 1

    global_optima = [0 for n in range(1,13)]

    curves, fitnesses, times = [], [], []

    for n in range(num_runs):
        c, f, t = run_and_plot(seed=np.random.randint(20), generate_plots=False)
        curves.append(c)
        fitnesses.append(f)
        times.append(t)


    ga_average = np.average([fitnesses[n]['GA'] for n in range(num_runs)], 0)
    sa_average = np.average([fitnesses[n]['SA'] for n in range(num_runs)], 0)
    rhc_average = np.average([fitnesses[n]['RHC'] for n in range(num_runs)], 0)
    mimic_average = np.average([fitnesses[n]['MIMIC'] for n in range(num_runs)], 0)

    lengths = [5*n for n in range(1,13)]

    # Length vs Fitness
    plt.plot(lengths, global_optima[:len(lengths)], label='Upper Bound', color='gold', linewidth=3, alpha=0.5)
    plt.plot(lengths, -1*np.array(rhc_average), label='RHC', color='mediumpurple', marker='*')
    plt.plot(lengths, -1*np.array(ga_average), label='GA', color='yellowgreen', marker='o')
    plt.plot(lengths, -1*np.array(sa_average), label='SA', color='cadetblue', marker=6)
    plt.plot(lengths, -1*np.array(mimic_average), label='MIMIC', color='lightpink')
    plt.legend(loc='best')
    plt.xlabel('Problem Length (bits)')
    plt.ylabel('Fitness')
    plt.title('Length vs. Average Fitness on Max K Color')
    plt.savefig('length_v_avg_fitness_k_color.png')
    # plt.show()
    plt.close()

    print('Done.')


# for generating average fitness
generate_average_fitness_plot()

# for generating graphs associated with one run
_, _, _ = run_and_plot()

# for saving excel files with mimic testing
hyperparam_test(np.random.randint(20))