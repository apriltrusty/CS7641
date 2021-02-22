import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import rc
from time import perf_counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, make_scorer, plot_confusion_matrix

rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})

def import_telescope_data():
    columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 
               'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

    data = read_csv('Data/magic04.data', names=columns)

    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,-1]

    return X, y


def print_gridsearch_results(clf, parameters, X_train, y_train):
    fhalf_scorer = make_scorer(fbeta_score, beta=0.5)
    
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, scoring=fhalf_scorer)
    grid_search.fit(X_train, y_train)

    # source for this function: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    print('Best parameters set found on development set:')
    print()
    print(grid_search.best_params_)
    print()
    print('Grid scores on development set:')
    print()
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r'
            % (mean, std * 2, params))
    print()

    exit(0)


def generate_learning_curve(clf, X_train, y_train, clf_name, clf_params, score_label):
    fhalf_scorer = make_scorer(fbeta_score, beta=0.5)
    
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, scoring=fhalf_scorer, random_state=1)
    
    # plotting code mainly sourced from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    # also add this citation in readme, not in report
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')
    plt.legend(loc='best')
    plt.ylabel(score_label.capitalize())
    plt.xlabel('Training set size')
    plt.xticks(rotation=45)
    plt.title('Learning Curves for ' + clf_name + ' with ' + clf_params)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot fit_time vs score
    # plt.grid()
    # plt.plot(fit_times_mean, test_scores_mean, 'o-')
    # plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # plt.xlabel('Seconds to fit')
    # plt.xticks(rotation=45)
    # plt.ylabel('Score')
    # plt.title('Performance of the ' + clf_name + ' model\nwith ' + str(clf_params))
    # plt.tight_layout()
    # plt.show()
    # plt.close()


def generate_complexity_curve(clf, clf_name, test_param, param_values, param_label, X_train, y_train, score_label):
    fhalf_scorer = make_scorer(fbeta_score, beta=0.5)
    
    train_scores, test_scores = validation_curve(
        clf, X_train, y_train, param_name=test_param, param_range=param_values, scoring=fhalf_scorer, cv=StratifiedKFold())
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(clf_name + ' Model Complexity Curves')
    plt.xlabel(param_label)
    plt.xticks(rotation=45)
    plt.ylabel(score_label.capitalize())
    lw = 2
    plt.plot(param_values, train_scores_mean, label='Training score', color='darkorange', lw=lw)
    plt.fill_between(param_values, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                    color='darkorange', lw=lw)
    plt.plot(param_values, test_scores_mean, label='Cross-validation score', color='navy', lw=lw)
    plt.fill_between(param_values, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                    color='navy', lw=lw)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.close()


def display_confusion_matrix(clf, X, y):
    X_train_subset, X_validation, y_train_subset, y_validation = train_test_split(X, y, random_state=0, shuffle=True, stratify=y)
    clf.fit(X_train_subset, y_train_subset)
    plot_confusion_matrix(clf, X_validation, y_validation, normalize='true')
    plt.show()
    plt.close()


def run_decision_tree_experiments(X_train, y_train, do_gridsearch=False):
    clf_name = 'Decision Tree'

    if do_gridsearch:
        gs_params = {'max_depth': [(5*n) for n in range(1,11)],
                 'ccp_alpha': [(0.0005*n) for n in range(11)],
                 'criterion': ['gini', 'entropy']}

        print_gridsearch_results(clf=DecisionTreeClassifier(), parameters=gs_params, X_train=X_train, y_train=y_train)

    generate_complexity_curve(clf=DecisionTreeClassifier(criterion='gini'), clf_name=clf_name,
                              test_param='ccp_alpha', param_values=np.array([(0.0001*n) for n in range(16)]),
                              param_label=r'$\alpha$', X_train=X_train, y_train=y_train, score_label='F-0.5 Score')

    generate_learning_curve(clf=DecisionTreeClassifier(ccp_alpha=0.0003, criterion='gini'), 
                            X_train=X_train, y_train=y_train, clf_name=clf_name, 
                            clf_params=r'$\alpha$=0.0003, split on Gini impurity', score_label='F-0.5 Score')

    # generate graph to show the depth of the tree (i.e., complexity of the model) compared to alpha
    # the lower alpha, the more complex the fitted model is, and the more prone to overfitting

    # models = []
    # test_alphas = np.array([(0.0001*n) for n in range(16)])

    # for alpha in test_alphas:
    #     model = DecisionTreeClassifier(ccp_alpha=alpha, criterion='gini')
    #     model.fit(X_train, y_train)
    #     models.append(model)

    # depths = [model.get_depth() for model in models]

    # plt.plot(test_alphas, np.array(depths), color='orchid')
    # plt.ylabel('Depth')
    # plt.xlabel(r'$\alpha$')
    # plt.title('Pruning aggresiveness vs. tree depth')
    # plt.show()
    # plt.close()

    # clf = DecisionTreeClassifier(ccp_alpha=0.0003)
    # display_confusion_matrix(clf=clf, X=X_train, y=y_train)


def run_boosted_tree_experiment(X_train, y_train, do_gridsearch=False):
    clf_name = 'AdaBoosted Decision Trees'

    if do_gridsearch:
        gs_params = {'n_estimators': [50*n for n in range(1,16)],
                 'learning_rate': [0.01, 0.1, 1],
                 'base_estimator__max_depth': [1,2]}

        print_gridsearch_results(clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), 
                                 parameters=gs_params, X_train=X_train, y_train=y_train)

    generate_learning_curve(clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion='gini'), 
                            n_estimators=300, learning_rate=1.0), X_train=X_train, y_train=y_train, clf_name=clf_name, 
                            clf_params='300 estimators, 1.0 learning rate', score_label='F-0.5 Score')
    

    
    # generate_complexity_curve(clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion='gini'), learning_rate=1.0), 
    #                           clf_name=clf_name, test_param='n_estimators', param_values=np.array([800 + 100*n for n in range(13)]), 
    #                           param_label='Number of Classifiers', X_train=X_train, y_train=y_train, score_label='F-0.5 Score')


    # special graph for boosted trees - two complexity curves in one
    fhalf_scorer = make_scorer(fbeta_score, beta=0.5)
    
    clf_1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion='gini'), learning_rate=1.0, random_state=0)
    clf_2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion='gini'), learning_rate=0.1, random_state=0)
    test_param = 'n_estimators'
    param_values = np.array([50*n for n in range(1,16)])


    train_scores_1, test_scores_1 = validation_curve(
        clf_1, X_train, y_train, param_name=test_param, param_range=param_values, scoring=fhalf_scorer, cv=StratifiedKFold())
    
    train_scores_2, test_scores_2 = validation_curve(
        clf_2, X_train, y_train, param_name=test_param, param_range=param_values, scoring=fhalf_scorer, cv=StratifiedKFold())
    
    train_scores_mean_1 = np.mean(train_scores_1, axis=1)
    train_scores_std_1 = np.std(train_scores_1, axis=1)
    test_scores_mean_1 = np.mean(test_scores_1, axis=1)
    test_scores_std_1 = np.std(test_scores_1, axis=1)

    train_scores_mean_2 = np.mean(train_scores_2, axis=1)
    train_scores_std_2 = np.std(train_scores_2, axis=1)
    test_scores_mean_2 = np.mean(test_scores_2, axis=1)
    test_scores_std_2 = np.std(test_scores_2, axis=1)

    plt.title(clf_name + ' Model Complexity Curves')
    plt.xlabel('Number of Classifiers')
    plt.xticks(rotation=45)
    plt.ylabel('F-0.5 Score')
    lw = 2
    
    # plot for classifier 1 (lr=1.0)
    plt.plot(param_values, train_scores_mean_1, label='Training score (lr=1.0)', color='darkorange', lw=lw)
    plt.fill_between(param_values, train_scores_mean_1 - train_scores_std_1, train_scores_mean_1 + train_scores_std_1, alpha=0.2,
                    color='darkorange', lw=lw)
    plt.plot(param_values, test_scores_mean_1, label='Cross-validation score (lr=1.0)', color='navy', lw=lw)
    plt.fill_between(param_values, test_scores_mean_1 - test_scores_std_1, test_scores_mean_1 + test_scores_std_1, alpha=0.2,
                    color='navy', lw=lw)
    
    # plot for classifier 2 (lr=0.1)
    plt.plot(param_values, train_scores_mean_2, label='Training score (lr=0.1)', color='gold', lw=lw)
    plt.fill_between(param_values, train_scores_mean_2 - train_scores_std_2, train_scores_mean_2 + train_scores_std_2, alpha=0.2,
                    color='gold', lw=lw)
    plt.plot(param_values, test_scores_mean_2, label='Cross-validation score (lr=0.1)', color='cornflowerblue', lw=lw)
    plt.fill_between(param_values, test_scores_mean_2 - test_scores_std_2, test_scores_mean_2 + test_scores_std_2, alpha=0.2,
                    color='cornflowerblue', lw=lw)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.close()


def run_knn_experiment(X_train, y_train, do_gridsearch=False):
    clf_name = 'K-Nearest Neighbors'

    if do_gridsearch:
        gs_params = {'n_neighbors': [n for n in range(1,10)],
                  'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                  'weights': ['uniform', 'distance']}

        print_gridsearch_results(clf=KNeighborsClassifier(), parameters=gs_params, X_train=X_train, y_train=y_train)

    generate_learning_curve(clf=KNeighborsClassifier(n_neighbors=10, weights='uniform'), 
                            X_train=X_train, y_train=y_train, clf_name=clf_name, 
                            clf_params='k=10, uniform weighted', score_label='F-0.5 Score')

    # double complexity curve
    scorer = make_scorer(fbeta_score, beta=0.5)

    clf_1 = KNeighborsClassifier(weights='uniform')
    clf_2 = KNeighborsClassifier(weights='distance')
    
    test_param = 'n_neighbors'
    param_values = np.array([1+2*n for n in range(1,21)])


    train_scores_1, test_scores_1 = validation_curve(
        clf_1, X_train, y_train, param_name=test_param, param_range=param_values, 
        scoring=scorer, cv=StratifiedKFold(shuffle=True))
    
    train_scores_2, test_scores_2 = validation_curve(
        clf_2, X_train, y_train, param_name=test_param, param_range=param_values, 
        scoring=scorer, cv=StratifiedKFold(shuffle=True))
    
    train_scores_mean_1 = np.mean(train_scores_1, axis=1)
    train_scores_std_1 = np.std(train_scores_1, axis=1)
    test_scores_mean_1 = np.mean(test_scores_1, axis=1)
    test_scores_std_1 = np.std(test_scores_1, axis=1)

    train_scores_mean_2 = np.mean(train_scores_2, axis=1)
    train_scores_std_2 = np.std(train_scores_2, axis=1)
    test_scores_mean_2 = np.mean(test_scores_2, axis=1)
    test_scores_std_2 = np.std(test_scores_2, axis=1)

    plt.title(clf_name + ' Model Complexity Curves')
    plt.xlabel('Number of Neighbors (k)')
    plt.xticks(rotation=45)
    plt.ylabel('F-0.5 Score')
    lw = 2
    
    # plot for classifier 1 (uniform)
    plt.plot(param_values, train_scores_mean_1, label='Training score (uniform weight)', color='darkorange', lw=lw)
    plt.fill_between(param_values, train_scores_mean_1 - train_scores_std_1, train_scores_mean_1 + train_scores_std_1, alpha=0.2,
                    color='darkorange', lw=lw)
    plt.plot(param_values, test_scores_mean_1, label='Cross-validation score (uniform weight)', color='navy', lw=lw)
    plt.fill_between(param_values, test_scores_mean_1 - test_scores_std_1, test_scores_mean_1 + test_scores_std_1, alpha=0.2,
                    color='navy', lw=lw)
    
    # plot for classifier 2 (distance)
    # plt.plot(param_values, train_scores_mean_2, label='Training score (distance weight)', color='gold', lw=lw)
    # plt.fill_between(param_values, train_scores_mean_2 - train_scores_std_2, train_scores_mean_2 + train_scores_std_2, alpha=0.2,
    #                 color='gold', lw=lw)
    plt.plot(param_values, test_scores_mean_2, label='Cross-validation score (distance weight)', color='cornflowerblue', lw=lw)
    plt.fill_between(param_values, test_scores_mean_2 - test_scores_std_2, test_scores_mean_2 + test_scores_std_2, alpha=0.2,
                    color='cornflowerblue', lw=lw)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.close()


def run_ann_experiment(X_train, y_train, do_gridsearch=False):
    clf_name = 'Artificial Neural Network'

    if do_gridsearch:
        gs_params = {'hidden_layer_sizes': [50,100,150,200,250,300],
                     'learning_rate_init': [0.001, 0.005, 0.01, 0.05],
                     'batch_size': [100,200,300]}

        print_gridsearch_results(clf=MLPClassifier(max_iter=600), parameters=gs_params, X_train=X_train, y_train=y_train)

    generate_learning_curve(clf=MLPClassifier(batch_size=100, hidden_layer_sizes=(150), learning_rate_init=0.003, max_iter=550, 
                                              random_state=0), 
                            X_train=X_train, y_train=y_train, clf_name=clf_name, 
                            clf_params='\nbatch size=100, hidden nodes=(150), initial learning rate=0.003',
                            score_label='F-0.5 Score')

    # # loss curve (training) and accuracy curve (validation)
    clf = MLPClassifier(batch_size=100, hidden_layer_sizes=(150), learning_rate_init=0.003, \
                        max_iter=800, random_state=0, early_stopping=True, tol=1e-16, n_iter_no_change=160)
    clf.fit(X=X_train, y=y_train)

    # plotting code from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
    fig, ax1 = plt.subplots()

    color = 'orchid'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(clf.loss_curve_, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim(0.25, 0.45)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'orange'
    ax2.set_ylabel('Validation Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(clf.validation_scores_, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    moving_average = np.convolve(clf.validation_scores_, np.ones(10), 'valid') / 10
    ax2.plot(range(len(clf.validation_scores_) - len(moving_average), len(clf.validation_scores_)), moving_average, color='orangered')
    # ax2.set_ylim(0.84, 0.90)
 
    plt.title('Loss and Accuracy Curves for Neural Network')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    plt.close()

    # effect of number of nodes in single hidden layer
    generate_complexity_curve(clf=MLPClassifier(batch_size=100, max_iter=800, learning_rate_init=0.005, random_state=0), clf_name=clf_name, 
                              test_param='hidden_layer_sizes', param_values=np.array([30*n for n in range(1,9)]), param_label='Number of Nodes in Hidden Layer',
                              X_train=X_train, y_train=y_train, score_label='F-0.5 Score')

    # effect of learning rates
    generate_complexity_curve(clf=MLPClassifier(batch_size=100, max_iter=800, hidden_layer_sizes=(150), random_state=0, 
                                                early_stopping=True, tol=1e-16, n_iter_no_change=160), 
                                clf_name=clf_name, test_param='learning_rate_init', param_values=np.array([0.001*n for n in range(1,11)]), 
                                param_label='Initial Learning Rate', X_train=X_train, y_train=y_train, score_label='F-0.5 Score')


def run_svm_experiment(X_train, y_train, do_gridsearch=False):
    clf_name = 'Support Vector Machine'
    
    if do_gridsearch:
        gs_params = {'C': [0.1, 1, 10, 100],
                    'gamma': [0.0001, 0.001, 0.01, 0.1],
                    'kernel': ['RBF']}

        print_gridsearch_results(clf=svm.SVC(), parameters=gs_params, X_train=X_train, y_train=y_train)

    generate_learning_curve(clf=svm.SVC(kernel='RBF', C=100, gamma=0.1), 
                            X_train=X_train, y_train=y_train, clf_name=clf_name, 
                            clf_params='RBF kernel, C=100, gamma=0.1', score_label='F-0.5 Score')
    
    # generate_complexity_curve(clf=svm.SVC(kernel='rbf', gamma='auto'), clf_name=clf_name,
    #                           test_param='C', param_values=np.array([1,10,100]), param_label='Regularization Parameter (C)',
    #                           X_train=X_train, y_train=y_train, score_label='F-0.5 Score')


def run_final_model(clf, name, X_train, y_train, X_test, y_test):
    begin_train_time = perf_counter()
    clf.fit(X_train, y_train)
    end_train_time = perf_counter()

    train_time = end_train_time - begin_train_time

    begin_predict_time = perf_counter()
    train_predictions = np.squeeze(clf.predict(X_train))
    test_predictions = np.squeeze(clf.predict(X_test))
    end_predict_time = perf_counter()

    predict_time = end_predict_time - begin_predict_time
    
    train_score = fbeta_score(y_true=y_train, y_pred=train_predictions, beta=0.5)
    test_score = fbeta_score(y_true=y_test, y_pred=test_predictions, beta=0.5)

    print('\nGot: %.2f%% F1 on the test set and %.2f%% F1 on the train set for the %s model.' % (test_score*100, train_score*100, name))
    print(f'{name} model took {round(train_time,5)} sec to train and {round(predict_time,5)} sec to predict.\n')
    if name == 'ANN':
        print(f'{name} model took {len(clf.loss_curve_)} iterations to train.\n')


def run_experiments():
    X, y = import_telescope_data()
    
    # relabel g (signal) as 1, h (background) as 0
    y = y.replace(to_replace='g', value=1).replace(to_replace='h', value=0)

    # split into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True, stratify=y)

    # standardize the X data, using the mean and std of the training data to standardize the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # experiments to determine best hyperparameters for various classifiers
    run_decision_tree_experiments(X_train=X_train, y_train=y_train, do_gridsearch=False)
    run_ann_experiment(X_train=X_train, y_train=y_train, do_gridsearch=False)
    run_boosted_tree_experiment(X_train=X_train, y_train=y_train, do_gridsearch=False)
    run_svm_experiment(X_train=X_train, y_train=y_train, do_gridsearch=False)
    run_knn_experiment(X_train=X_train, y_train=y_train, do_gridsearch=True)


    # now, train models with determined hyperparameters and see the results on the train/test sets!
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.0003, criterion='gini', random_state=0)
    run_final_model(clf=dt_clf, name='Decision Tree', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    adaboost_clf = AdaBoostClassifier(n_estimators=300, learning_rate=1, random_state=0)
    run_final_model(clf=adaboost_clf, name='Boosted Tree', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    ann_clf = MLPClassifier(batch_size=100, hidden_layer_sizes=(150), learning_rate_init=0.003, max_iter=550, random_state=0)
    run_final_model(clf=ann_clf, name='ANN', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    svm_clf = svm.SVC(kernel='rbf', C=100, gamma=0.1)
    run_final_model(clf=svm_clf, name='SVM', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    knn_clf = KNeighborsClassifier(n_neighbors=10, weights='uniform')
    run_final_model(clf=knn_clf, name='KNN', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    



run_experiments()