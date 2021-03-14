from mlrose_hiive.neural import NeuralNetwork
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter


def import_telescope_data():
    columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 
               'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

    data = read_csv('/Users/apriltrusty/Documents/OMSCS/Machine Learning/Datasets/Magic/magic04.data', names=columns)

    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,-1]

    return X, y


def train_and_report(nn, X_train, X_test, y_train, y_test, name):
    start_time = perf_counter()
    
    nn.fit(X_train, y_train)

    y_train_pred = nn.predict(X_train)
    y_train_score = fbeta_score(y_train, y_train_pred, 0.5)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn.predict(X_test)
    y_test_score = fbeta_score(y_test, y_test_pred, 0.5)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    
    end_time = perf_counter()

    print(f'{name} Train: {y_train_accuracy}. Test: {y_test_accuracy}')
    
    if name == 'Gradient Descent': factor = -1
    else: factor = 1

    time_taken = end_time - start_time
    curve = factor * np.array(nn.fitness_curve)

    return curve, time_taken, [y_train_accuracy, y_test_accuracy]


def experiment_all_algorithms(seed=1, plotting=False):
    X, y = import_telescope_data()

    # relabel g (signal) as 1, h (background) as 0
    y = y.replace(to_replace='g', value=1).replace(to_replace='h', value=0)
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, train_size=0.1, stratify=y, random_state=seed)

    # standardize the X data, using the mean and std of the training data to standardize the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize neural network and fit
    nn_gd = NeuralNetwork(hidden_nodes=[10], activation='relu', algorithm='gradient_descent', max_iters=2500, 
                            bias=True, is_classifier=True, learning_rate=0.001, early_stopping=True, clip_max=5,
                            max_attempts=300, random_state=seed, curve=True)

    nn_ga = NeuralNetwork(hidden_nodes=[10], activation='relu', algorithm='genetic_alg', pop_size=100, 
                            mutation_prob=0.1, max_iters=1000, bias=True, is_classifier=True, clip_max=2,
                            learning_rate=0.1, early_stopping=True, max_attempts=100, random_state=seed, curve=True)
    
    nn_rhc = NeuralNetwork(hidden_nodes=[10], activation='relu', algorithm='random_hill_climb', max_iters=2000, clip_max=5,
                            bias=True, is_classifier=True, learning_rate=0.2, early_stopping=True, restarts=10,
                            max_attempts=120, random_state=seed, curve=True)
    
    nn_sa = NeuralNetwork(hidden_nodes=[10], activation='relu', algorithm='simulated_annealing', max_iters=7000, clip_max=5,
                            bias=True, is_classifier=True, learning_rate=0.1, early_stopping=True,
                            max_attempts=120, random_state=seed, curve=True)

    sa_curve, sa_time, sa_score = train_and_report(nn=nn_sa, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name='Simulated Annealing')
    ga_curve, ga_time, ga_score = train_and_report(nn=nn_ga, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name='Genetic Algorithm')
    rhc_curve, rhc_time, rhc_score = train_and_report(nn=nn_rhc, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name='Randomized Hill Climbing')
    gd_curve, gd_time, gd_score = train_and_report(nn=nn_gd, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name='Gradient Descent')

    if plotting:
        # plotting NN curves 
        plt.plot(range(len(sa_curve)), sa_curve, label='SA', color='cadetblue')  
        plt.plot(range(len(rhc_curve)), rhc_curve, label='RHC', color='mediumpurple') 
        plt.plot(range(len(gd_curve)), gd_curve, label='GD', color='gold')
        plt.plot(range(len(ga_curve)), ga_curve, label='GA', color='yellowgreen', linestyle='dashed')
        plt.title('Iteration vs. Loss') 
        plt.legend(loc='best')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.savefig('nn_curves.png') 
        plt.close()

    return (sa_curve, sa_time, sa_score), (ga_curve, ga_time, ga_score), (rhc_curve, rhc_time, rhc_score), (gd_curve, gd_time, gd_score)


def print_averages():
    np.random.seed(1)
    num_runs = 5

    sa_curves = []
    sa_times = []
    sa_scores = []

    ga_curves = []
    ga_times = []
    ga_scores = []

    rhc_curves = []
    rhc_times = []
    rhc_scores = []

    gd_curves = []
    gd_times = []
    gd_scores = []

    for n in range(num_runs):
        sa, ga, rhc, gd = experiment_all_algorithms(seed=np.random.randint(20))
        
        sa_curves.append(sa[0])
        sa_times.append(sa[1])
        sa_scores.append(sa[2])

        ga_curves.append(ga[0])
        ga_times.append(ga[1])
        ga_scores.append(ga[2])

        rhc_curves.append(rhc[0])
        rhc_times.append(rhc[1])
        rhc_scores.append(rhc[2])

        gd_curves.append(gd[0])
        gd_times.append(gd[1])
        gd_scores.append(gd[2])


    print(f'SA average training: {np.average([s[0] for s in sa_scores])} ({np.std([s[0] for s in sa_scores])})')
    print(f'SA average testing: {np.average([s[1] for s in sa_scores])} ({np.std([s[1] for s in sa_scores])})')
    print(f'SA average time: {np.average(sa_times)} ({np.std(sa_times)})')

    print(f'GA average training: {np.average([s[0] for s in ga_scores])} ({np.std([s[0] for s in ga_scores])})')
    print(f'GA average testing: {np.average([s[1] for s in ga_scores])} ({np.std([s[1] for s in ga_scores])})')
    print(f'GA average time: {np.average(ga_times)} ({np.std(ga_times)})')

    print(f'RHC average training: {np.average([s[0] for s in rhc_scores])} ({np.std([s[0] for s in rhc_scores])})')
    print(f'RHC average testing: {np.average([s[1] for s in rhc_scores])} ({np.std([s[1] for s in rhc_scores])})')
    print(f'RHC average time: {np.average(rhc_times)} ({np.std(rhc_times)})')
    
    print(f'GD average training: {np.average([s[0] for s in gd_scores])} ({np.std([s[0] for s in gd_scores])})')
    print(f'GD average testing: {np.average([s[1] for s in gd_scores])} ({np.std([s[1] for s in gd_scores])})')
    print(f'GD average time: {np.average(gd_times)} ({np.std(gd_times)})')


_, _, _, _ = experiment_all_algorithms(plotting=True)
print_averages()