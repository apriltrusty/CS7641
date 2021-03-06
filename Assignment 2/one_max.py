import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_annealing_comparison():
    seed = 1
    np.random.seed(seed)
    length = 100

    fitness = mlrose_hiive.OneMax()
    problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness)

    _, _, sa_curve_high_fast = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(init_temp=1e1, decay=0.99), max_attempts=200, random_state=1, curve=True)
    _, _, sa_curve_low_fast = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(init_temp=1, decay=0.99), max_attempts=200, random_state=1, curve=True)
    _, _, sa_curve_high_slow = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(init_temp=1e1, decay=0.999), max_attempts=200, random_state=1, curve=True)
    _, _, sa_curve_low_slow = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(init_temp=1, decay=0.999), max_attempts=200, random_state=1, curve=True)

    plt.plot(range(len(sa_curve_high_slow)), sa_curve_high_slow, label=r'$T_{0}=10,$ decay rate=0.999', color='crimson', linewidth=0.8)
    plt.plot(range(len(sa_curve_high_fast)), sa_curve_high_fast, label=r'$T_{0}=10,$ decay rate=0.99', color='darkorange', linewidth=0.8)
    plt.plot(range(len(sa_curve_low_slow)), sa_curve_low_slow, label=r'$T_{0}=1,$ decay rate=0.999', color='slateblue', linewidth=0.8)
    plt.plot(range(len(sa_curve_low_fast)), sa_curve_low_fast, label=r'$T_{0}=1,$ decay rate=0.99', color='mediumturquoise', linewidth=0.8)
    plt.hlines(length, xmin=0, xmax=max(len(sa_curve_high_fast), len(sa_curve_low_fast), len(sa_curve_high_slow), len(sa_curve_low_slow)), linestyles='dashed', label='Global Optimum', color='gold')
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title(f'Iteration vs. Fitness for One Max of Length {length}\nwith Simulated Annealing, Geometric Decay')
    plt.savefig('iteration_v_fitness_one_max_annealing.png')
    # plt.show()
    plt.close()


def run_and_plot(seed=1, generate_plots=True):
    lengths = [10*n for n in range(1,16)] # 10, 20, ..., 150

    global_optima = lengths.copy()

    iteration_list = [n for n in range(1,1001)] # iterations at which to log results.
    fixed_length = 100 # for this length, plot iteration versus fitness

    # SA parameters
    temperature_list = [0.7] # initial temperature -- higher means more exploration
    decay_list = [mlrose_hiive.GeomDecay] # slower decay means more exploration. exp decays slower than geom.

    # GA parameters
    population_sizes = [100]
    mutation_rates = [0.2]

    # RHC parameters
    restart_list = [0]

    # MIMIC parameters
    population_sizes = [200]
    keep_percent_list = [0.2]

    # log best fitnesses acheived for each algorithm, to plot later
    best_fitnesses = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}
    
    # log time taken for each algorithm, to plot later
    times = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}
    fevals = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}
    
    # fitness score development for set length
    fitness_development = {'SA': [], 'GA': [], 'RHC': [], 'MIMIC': []}

    for length in lengths:
        print(f'Starting length {length}')
        fitness = mlrose_hiive.OneMax()
        problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness)

        runners = [mlrose_hiive.SARunner(problem=problem, experiment_name='SA', seed=seed, iteration_list=iteration_list, 
                                         temperature_list=temperature_list, decay_list=decay_list, max_attempts=200),

                   mlrose_hiive.GARunner(problem=problem, experiment_name='GA', seed=seed, iteration_list=iteration_list, 
                                            population_sizes=population_sizes, mutation_rates=mutation_rates, max_attempts=200),

                   mlrose_hiive.RHCRunner(problem=problem, experiment_name='RHC', seed=seed, iteration_list=iteration_list, 
                                            restart_list=restart_list, max_attempts=200),

                   mlrose_hiive.MIMICRunner(problem=problem, experiment_name='MIMIC', seed=seed, iteration_list=iteration_list, 
                                                population_sizes=population_sizes, keep_percent_list=keep_percent_list, 
                                                use_fast_mimic=True, max_attempts=200)]
        
        for runner in runners:
            run_stats, curves = runner.run()

            # compiling length versus max fitness achieved plot
            best_fitnesses[runner._experiment_name].append(max(curves['Fitness']))

            # compiling length versus clock time plot
            times[runner._experiment_name].append(curves['Time'].iloc[-1] - curves['Time'].iloc[0])

            # fevals
            eval_count_RHC = sum(curves['FEvals'].iloc[find_peaks(curves['FEvals'])[0]])
            eval_count_others = curves['FEvals'].iloc[-1]
            eval_count = max(eval_count_RHC, eval_count_others)
            fevals[runner._experiment_name].append(eval_count)

            # plot iteration versus fitness for the set length
            if length == fixed_length:
                fitness_development[runner._experiment_name] = curves['Fitness']


    if generate_plots:
        # Iteration vs Fitness
        plt.plot(range(len(fitness_development['RHC'])), fitness_development['RHC'], label='RHC', color='mediumpurple')
        plt.plot(range(len(fitness_development['SA'])), fitness_development['SA'], label='SA', color='cadetblue')
        plt.plot(range(len(fitness_development['GA'])), fitness_development['GA'], label='GA', color='yellowgreen')
        plt.plot(range(len(fitness_development['MIMIC'])), fitness_development['MIMIC'], label='MIMIC', color='lightpink')
        plt.hlines(global_optima[int(fixed_length/10) - 1], xmin=0, xmax=max(len(fitness_development['SA']),len(fitness_development['GA']), len(fitness_development['RHC']), len(fitness_development['MIMIC'])), 
                        linestyles='dashed', label='Global Optimum', color='gold')
        plt.legend(loc='best')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title(f'Iteration vs. Fitness for One Max of Length {fixed_length}')
        plt.savefig('iteration_v_fitness_one_max.png')
        # plt.show()
        plt.close()

        # Length vs Fitness
        plt.plot(lengths, global_optima[:len(lengths)], label='Global Optimum', color='gold', linestyle='dashed', marker='o')
        plt.plot(lengths, lengths, label='Local Optimum', color='lightgrey', linestyle='dashed', marker='o')
        plt.plot(lengths, best_fitnesses['RHC'], label='RHC', color='mediumpurple')
        plt.plot(lengths, best_fitnesses['SA'], label='SA', color='cadetblue')
        plt.plot(lengths, best_fitnesses['GA'], label='GA', color='yellowgreen')
        plt.plot(lengths, best_fitnesses['MIMIC'], label='MIMIC', color='lightpink')
        plt.legend(loc='best')
        plt.xlabel('Problem Length (bits)')
        plt.ylabel('Fitness')
        plt.title('Length vs. Fitness on One Max')
        plt.savefig('length_v_fitness_one_max.png')
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
        plt.title('Length vs. Time on One Max')
        plt.savefig('length_v_time_one_max.png')
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
        plt.title('Length vs. Fitness Function Evaluations on One Max')
        plt.savefig('length_v_fevals_one_max.png')
        # plt.show()
        plt.close()

        print('Done')

    return fitness_development, best_fitnesses, times


def generate_average_fitness_plot():
    np.random.seed(1)
    num_runs = 5

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

    lengths = [10*n for n in range(1,16)]
    global_optima = lengths.copy()

    # Length vs Fitness
    plt.plot(lengths, global_optima[:len(lengths)], label='Global Optimum', color='gold', marker='o')
    plt.plot(lengths, sa_average, label='SA', color='cadetblue')
    plt.plot(lengths, ga_average, label='GA', color='yellowgreen')
    plt.plot(lengths, rhc_average, label='RHC', color='mediumpurple')
    plt.plot(lengths, mimic_average, label='MIMIC', color='lightpink')
    plt.legend(loc='best')
    plt.xlabel('Problem Length (bits)')
    plt.ylabel('Fitness')
    plt.title('Length vs. Average Fitness on One Max')
    plt.savefig('length_v_avg_fitness_one_max.png')
    # plt.show()
    plt.close()

    print('Done.')   


generate_average_fitness_plot()
_, _, _ = run_and_plot()
plot_annealing_comparison()