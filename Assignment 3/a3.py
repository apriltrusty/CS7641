import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.stats import kurtosis
import seaborn as sns

import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score, make_scorer, f1_score
from sklearn.manifold import TSNE

from sklearn.decomposition import FastICA, PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection as RP

from yellowbrick.features import Rank2D
from kneed import KneeLocator

"""If you're asking about cluster or dim. red. experiments, there are really two phases... 
one in which you pick K clusters or N components. 

The other is "validating" that optimal # and building more intuition after running the algorithm... 
so looking at the quality of the clusters, seeing how your features contribute to your clusters, 
visualizing these using various techniques (up to you). 

For dim. red., you can look at what your projection looks like, try to see how your projection 
matches up with the original features, or introduce some other metric... whatever makes sense."""

# width, height
m = 1.6/2.33
width = 4
plt.rcParams['figure.figsize'] = (width,width*m)
linewidth = 1

contrasting_colors = ['crimson','lawngreen','gold','turquoise','blueviolet','violet','darkorange','cornflowerblue','hotpink','peru']


def import_red_wine_data():
    data = pd.read_csv('Data/winequality-red.csv', sep=';')

    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,-1]

    # transform data to binary classes; 1 to 5 is low quality, 6-10 is high quality
    y = y.replace(to_replace=[1,2,3,4,5], value=0).replace([6,7,8,9,10], value=1)

    return data, X, y


def import_telescope_data():
    columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 
               'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

    data = pd.read_csv('Data/magic04.data', names=columns)
    data = data.replace(to_replace={'g': 1, 'h': 0})

    X = data.iloc[:,0:data.shape[1]-1]
    y = data.iloc[:,-1]

    return data, X, y


def split_and_scale(X, y, scaler=StandardScaler(), pct=0.5):
    # split into testing and training sets
    if X.columns[0] == 'fLength':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.022, train_size=0.066, 
                                                            shuffle=True, stratify=y, random_state=1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=pct*0.75, test_size=pct*0.25, 
                                                            shuffle=True, stratify=y, random_state=1)
    

    # standardize the X data, using the mean and std of the training data to standardize the test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def experiment_1(X, y, dataset_name, n_clusters_k_means=None, n_clusters_gmm=None, save_graphs=True, dim_red='No'):
    """Run the clustering algorithms on the datasets and describe what you see."""
    
    # based on: 
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    def eval_against_labels(X, y, estimator, estimator_name):    
        
        # metrics which use labels (cannot use for choosing number of clusters)
        clustering_metrics_supervised = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]

        t0 = perf_counter()
        estimator.fit(X)
        fit_time = perf_counter() - t0

        y_pred = estimator.predict(X)
        
        results = [fit_time]
        if estimator_name == 'K-Means': results += [estimator.inertia_]
        results += [m(y, y_pred) for m in clustering_metrics_supervised]

        # metrics which don't require labels
        results += [metrics.silhouette_score(X, y_pred, metric="euclidean", sample_size=300,)]
        results += [metrics.calinski_harabasz_score(X, y_pred)]
        results += [metrics.davies_bouldin_score(X, y_pred)]

        # Show the results
        print(f'\nResults for {estimator_name}:')
        print(82 * '_')

        for i in range(max(y_pred)+1):
            print('Cluster {}, Label 0: {}'.format(i, y[y_pred == i].shape[0] - y[y_pred == i].sum()))
            print('Cluster {}, Label 1: {}'.format(i, y[y_pred == i].sum()))

        return y_pred
    
    def plot_TSNE(X, cluster_labels, title=None, p=30):
        tsne = TSNE(n_components=2, random_state=1, perplexity=p, n_iter=2000)
        X_tsne = tsne.fit_transform(X)

        for cluster_id in range(max(cluster_labels)+1):
            plt.scatter(X_tsne[:,0][cluster_labels==cluster_id], X_tsne[:,1][cluster_labels==cluster_id], label=cluster_id, s=120, c=contrasting_colors[cluster_id], alpha=0.25)

        if title is not None: 
            # plt.title(title)
            plt.legend()
            plt.savefig(f'Graphs/{dataset_name}/Experiment {exp_num}/TSNE/{title}.png')
        # plt.show()
        plt.close()

    # determine what value of K to use for k-means
    fit_times = []
    silhouettes = []    # higher is better
    calinskis = []      # higher is better
    davies = []         # closer to 0 is better
    k_range = range(2,51)

    if dim_red == 'No': exp_num = 1
    else: exp_num = 3

    k_means_k = n_clusters_k_means[dim_red]
    gmm_k = n_clusters_gmm[dim_red]
    
    for k in k_range:
        # print(f'Testing k={k} for K-Means.')
        estimator = KMeans(n_clusters=k, random_state=1)

        t0 = perf_counter()
        estimator.fit(X)
        fit_time = perf_counter() - t0
        y_pred = estimator.predict(X)
        
        fit_times.append(fit_time)
        silhouettes.append(metrics.silhouette_score(X, y_pred, metric='euclidean', sample_size=300,))
        if dim_red == 'KPCA' and dataset_name == 'Telescope':
            calinskis.append(metrics.calinski_harabasz_score(X, y_pred)/5000)
        else:
            calinskis.append(metrics.calinski_harabasz_score(X, y_pred)/1000)
        davies.append(metrics.davies_bouldin_score(X, y_pred))
    
    
    if save_graphs:
        sil_color = 'navy'
        cal_color = 'orange'
        davies_color = 'green'

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.grid(axis='y')
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor=davies_color)
        ax2.grid(False)
        markers_on = [n_clusters_k_means[dim_red]-2]

        ax1.plot(k_range, silhouettes, 'o', ls='-', label='Silhouette Coefficient', color=sil_color, linewidth=linewidth, markevery=markers_on)
        if dim_red == 'KPCA' and dataset_name == 'Telescope':
            ax1.plot(k_range, calinskis, 'o', ls='-', label='Calinski-Harabasz Index/5000', color=cal_color, linewidth=linewidth, markevery=markers_on)
        else:
            ax1.plot(k_range, calinskis, 'o', ls='-', label='Calinski-Harabasz Index/1000', color=cal_color, linewidth=linewidth, markevery=markers_on)
        ax2.plot(k_range, davies, 'o', ls='-', label='Davies-Bouldin Index', color=davies_color, linewidth=linewidth, markevery=markers_on)

        # fig.legend(frameon=True, loc='center', bbox_to_anchor=(0.75, 0.75))
        title = f'K-Means Metrics for {dataset_name} Data\nwith {dim_red} Dimensionality Reduction'
        # plt.title(title)
        plt.tight_layout()
        plt.savefig(f'Graphs/{dataset_name}/Experiment {exp_num}/{title}.png')
        # plt.show()
        plt.close()
    

    for p in range(10,210,10):
        kmeans_clf = KMeans(n_clusters=k_means_k, random_state=1)
        kmeans_clf.fit(X)
        cluster_labels = kmeans_clf.predict(X) # cluster labels
        plot_TSNE(X, cluster_labels, title=f'TSNE (Perplexity {p}) for {dataset_name} Data\nClustered with K-Means to {k_means_k} Clusters, {dim_red} Dimensionality Reduction', p=p)
    
    
    # determine what value of k to use for GMM
    fit_times = []
    bics = []
    silhouettes = []    # higher is better
    calinskis = []      # higher is better
    davies = []         # closer to 0 is better
    
    for k in k_range:
        # print(f'Testing k={k} for GMM.')
        estimator = GaussianMixture(n_components=k, random_state=1, verbose=0)

        t0 = perf_counter()
        estimator.fit(X)
        fit_time = perf_counter() - t0
        y_pred = estimator.predict(X)
        
        fit_times.append(fit_time)
        bic = estimator.bic(X)
        bics.append(bic)
        silhouettes.append(metrics.silhouette_score(X, y_pred, metric='euclidean', sample_size=300,))
        calinskis.append(metrics.calinski_harabasz_score(X, y_pred))
        davies.append(metrics.davies_bouldin_score(X, y_pred))
    
    if save_graphs:
        bic_color = 'salmon'
        sil_color = 'navy'
        cal_color = 'orange'
        davies_color = 'green'

        # BIC and Silhouette
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.grid(axis='y')
        ax1.tick_params(axis='y', labelcolor=sil_color)
        # ax1.set_ylabel('Silhouette Coefficient', color=sil_color)
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor=bic_color)
        # ax2.set_ylabel('BIC', color=bic_color)
        ax2.grid(False)
        markers_on = [n_clusters_gmm[dim_red]-2]

        ax1.plot(k_range, silhouettes, 'o', ls='-', label='Silhouette Coefficient', color=sil_color, linewidth=linewidth, markevery=markers_on)
        ax2.plot(k_range, bics, 'o', ls='-', label='BIC', color=bic_color, linewidth=linewidth, markevery=markers_on)

        # fig.legend(frameon=True, loc='center', bbox_to_anchor=(0.75, 0.75))
        title = f'GMM BIC & Silhouette Coeff. for {dataset_name} Data\nwith {dim_red} Dimensionality Reduction'
        # plt.title(title)
        plt.tight_layout()
        plt.savefig(f'Graphs/{dataset_name}/Experiment {exp_num}/{title}.png')
        # plt.show()
        plt.close()

        # Calinski and Davies
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.grid(axis='y')
        ax1.tick_params(axis='y', labelcolor=cal_color)
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor=davies_color)
        ax2.grid(False)

        ax1.plot(k_range, calinskis, 'o', ls='-', label='Calinski-Harabasz Index', color=cal_color, linewidth=linewidth, markevery=markers_on)
        ax2.plot(k_range, davies, 'o', ls='-', label='Davies-Bouldin Index', color=davies_color, linewidth=linewidth, markevery=markers_on)

        # fig.legend(frameon=True, loc='center', bbox_to_anchor=(0.75, 0.75))
        title = f'GMM Calinski-Harabasz and Davies-Bouldin Indeces for {dataset_name} Data\nwith {dim_red} Dimensionality Reduction'
        # plt.title(title)
        plt.tight_layout()
        plt.savefig(f'Graphs/{dataset_name}/Experiment {exp_num}/{title}.png')
        # plt.show()
        plt.close()
    
    
    for p in range(10,210,10):
        gmm_clf = GaussianMixture(n_components=gmm_k, random_state=1, verbose=0)
        gmm_clf.fit(X)
        cluster_labels = gmm_clf.predict(X) # cluster labels
        plot_TSNE(X, cluster_labels, title=f'TSNE (Perplexity {p}) for {dataset_name} Data\nClustered with GMM to {gmm_k} Clusters, {dim_red} Dimensionality Reduction', p=p)
    
    # see how k-Means and GMM do when compared against labels!
    kmeans_clf = KMeans(n_clusters=k_means_k, random_state=1)
    labels_k = eval_against_labels(X=X, y=y, estimator=kmeans_clf, estimator_name='K-Means')

    centers = pd.DataFrame(kmeans_clf.cluster_centers_)
    with pd.ExcelWriter(f'{dataset_name}_{dim_red} reduction_kmeans_centers.xlsx') as writer:
                centers.to_excel(writer, sheet_name=f'{dim_red} Reduc. Centers')

    gmm_clf = GaussianMixture(n_components=gmm_k, random_state=1, verbose=0)
    labels_gmm = eval_against_labels(X=X, y=y, estimator=gmm_clf, estimator_name='Gaussian Mixture Model')

    centers = pd.DataFrame(gmm_clf.means_)
    with pd.ExcelWriter(f'{dataset_name}_{dim_red} reduction_gmm_centers.xlsx') as writer:
                centers.to_excel(writer, sheet_name=f'{dim_red} Reduc. Centers')

    return labels_k, labels_gmm


def experiment_2(X, y, dataset_name, n_features, save_graphs):
    """Apply the dimensionality reduction algorithms to the two datasets and describe what you see."""
    
    def plot_TSNE(X, title=None, p=30):
        tsne = TSNE(n_components=2, random_state=1, perplexity=p)
        X_tsne = tsne.fit_transform(X)

        plt.scatter(X_tsne[:,0][y==0], X_tsne[:,1][y==0], s=120, c=contrasting_colors[0], label='y=0', alpha=0.25)
        plt.scatter(X_tsne[:,0][y==1], X_tsne[:,1][y==1], s=120, c=contrasting_colors[1], label='y=1', alpha=0.25)

        if title is not None: 
            # plt.title(title)
            plt.legend()
            plt.savefig(f'Graphs/{dataset_name}/Experiment 2/TSNE/{title}.png')
        # plt.show()
        plt.close()

    n_components_range = range(1, X.shape[1]+1)

    n_ICA = n_features['ICA']
    n_PCA = n_features['PCA']
    n_RP = n_features['RP']
    n_KPCA = n_features['KPCA']
    
    # ICA
    avg_kurtoses = []
    max_kurtoses = []
    ica_re = []

    for n_components in n_components_range:
        ica = FastICA(n_components=n_components, max_iter=10000, random_state=1, tol=1e-3)
        transformed_X = ica.fit_transform(X)
        avg_kurtosis = np.average(kurtosis(transformed_X, axis=0))
        avg_kurtoses.append(avg_kurtosis)
        max_kurtoses.append(max(kurtosis(transformed_X, axis=0)))

        inverse_X = np.linalg.pinv((ica.components_.T))
        reconstructed_data = np.dot(transformed_X,(inverse_X))
        ica_re.append(np.sum(abs(X - reconstructed_data), axis=None) / X.size)
    
    if save_graphs:
        
        # plot kurtosis
        markers_on = [n_ICA-1]
        plt.plot(n_components_range, avg_kurtoses, 'o', ls='-', color='olive', label='Average Kurtosis', markevery=markers_on)
        plt.plot(n_components_range, max_kurtoses, 'o', ls='-', color='palevioletred', label='Maximum Kurtosis', markevery=markers_on)
        plt.xlabel('Number of Components (n)')
        title = f'ICA Kurtosis for {dataset_name} Data'
        # plt.title(title)
        plt.legend(loc='best')
        plt.savefig(f'Graphs/{dataset_name}/Experiment 2/{title}.png')
        # plt.show()
        plt.close()
        
        # plot TSNE
        ica = FastICA(n_components=n_ICA, random_state=1)
        transformed_X = ica.fit_transform(X)
        
        # for p in range(10,210,10):
        #     plot_TSNE(transformed_X, title=f'TSNE (Perplexity {p}) for {dataset_name} Data\nReduced by ICA to {n_ICA} Components', p=p)
        
        sns.pairplot(pd.DataFrame(np.hstack((transformed_X,np.atleast_2d(y).T))), hue=n_ICA, x_vars=[3], y_vars=[7])
        plt.savefig(f'Graphs/{dataset_name}/Experiment 2/Pairplot for ICA with {n_ICA}.png')
        plt.close()
        

    # PCA
    sum_eigenvalues = []
    avg_eigenvalues = []
    std_eigenvalues = []
    pca_re = []

    for n_components in n_components_range:
        pca = PCA(n_components=n_components, random_state=1)
        pca.fit(X)
        sum_eigenvalues.append(sum(pca.explained_variance_))
        avg_eigenvalues.append(np.average(pca.explained_variance_))
        std_eigenvalues.append(np.std(pca.explained_variance_))

        transformed_X = pca.fit_transform(X)
        inverse_X = np.linalg.pinv((pca.components_.T))
        reconstructed_data = np.dot(transformed_X,(inverse_X))
        pca_re.append(np.sum(abs(X - reconstructed_data), axis=None) / X.size)

    
    if save_graphs:
        # plot eigenvalues
        markers_on = [n_PCA-1]
        plt.fill_between(n_components_range, np.array(avg_eigenvalues) - np.array(std_eigenvalues),
                         np.array(avg_eigenvalues) + np.array(std_eigenvalues), alpha=0.25,
                         color='mediumpurple')
        plt.plot(n_components_range, avg_eigenvalues, 'o', ls='-', color='mediumpurple', label='Average of Eigenvalues', markevery=markers_on)
        plt.plot(n_components_range, sum_eigenvalues, 'o', ls='-', color='orange', label='Sum of Eigenvalues', markevery=markers_on)
        plt.xlabel('Number of Components (n)')
        title = f'PCA Eigenvalues for {dataset_name} Data'
        # plt.title(title)
        plt.legend(loc='best')
        plt.savefig(f'Graphs/{dataset_name}/Experiment 2/{title}.png')
        # plt.show()
        plt.close()
        
        # plot TSNE
        pca = PCA(n_components=n_PCA, random_state=1)
        transformed_X = pca.fit_transform(X)

        # for p in range(10,70,10):
        #     plot_TSNE(transformed_X, title=f'TSNE (Perplexity {p}) for {dataset_name} Data\nReduced by PCA to {n_PCA} Components', p=p)
        
        sns.pairplot(pd.DataFrame(np.hstack((transformed_X,np.atleast_2d(y).T))), hue=n_PCA, x_vars=[0], y_vars=[1])
        plt.savefig(f'Graphs/{dataset_name}/Experiment 2/Pairplot for PCA with {n_PCA}.png')
        # plt.show()
        plt.close()
    

    # RP
    reconstruction_errors = []
    for run in range(10):
        r = []
        for n_components in n_components_range:
            rp = RP(n_components=n_components)
            transformed_X = rp.fit_transform(X)
            inverse_X = np.linalg.pinv((rp.components_.T))
            reconstructed_data = np.dot(transformed_X,(inverse_X))
            r.append(np.sum(abs(X - reconstructed_data), axis=None) / X.size)
        reconstruction_errors.append(r)
    
    if save_graphs:
        
        markers_on = [n_RP-1]
        for run in range(10):
            plt.plot(n_components_range, reconstruction_errors[run], 'o', ls='-', color=contrasting_colors[run], markevery=markers_on)
        plt.xlabel('Number of Components (n)')
        title = f'RP Mean Absolute Reconstruction Error for {dataset_name} Data'
        # plt.title(title)
        # plt.legend(loc='best')
        plt.savefig(f'Graphs/{dataset_name}/Experiment 2/{title}.png')
        # plt.show()
        plt.close()
        
        # plot TSNE
        rp = RP(n_components=n_RP, random_state=1)
        transformed_X = rp.fit_transform(X)
        
        # for p in range(10,210,10):
        #     plot_TSNE(transformed_X, title=f'TSNE (Perplexity {p}) for {dataset_name} Data\nReduced by RP to {n_RP} Components', p=p)
        
        sns.pairplot(pd.DataFrame(np.hstack((transformed_X,np.atleast_2d(y).T))), hue=n_RP, x_vars=[1], y_vars=[5])
        plt.savefig(f'Graphs/{dataset_name}/Experiment 2/Pairplot for RP with {n_RP}.png')
        plt.close()
    
    # Kernel PCA
    for kernel in ['rbf']:
        
        std_eigenvalues = []
        avg_eigenvalues = []
        sum_eigenvalues = []
        
        for n_components in n_components_range:
            kpca = KernelPCA(n_components=n_components, kernel=kernel, random_state=1)
            kpca.fit(X)
            std_eigenvalues.append(np.std(kpca.lambdas_))
            avg_eigenvalues.append(np.average(kpca.lambdas_))
            sum_eigenvalues.append(sum(kpca.lambdas_))
        
        if save_graphs:
            markers_on = [n_KPCA-1]
            plt.fill_between(n_components_range, np.array(avg_eigenvalues) - np.array(std_eigenvalues),
                         np.array(avg_eigenvalues) + np.array(std_eigenvalues), alpha=0.25,
                         color='mediumpurple')
            plt.plot(n_components_range, avg_eigenvalues, 'o', ls='-', color='mediumpurple', label='Average of Eigenvalues', markevery=markers_on)
            plt.plot(n_components_range, sum_eigenvalues, 'o', ls='-', color='orange', label='Sum of Eigenvalues', markevery=markers_on)
            plt.xlabel('Number of Components (n)')
            title = f'{kernel.capitalize()}-Kernel KPCA Eigenvalues for {dataset_name} Data'
            # plt.title(title)
            plt.legend(loc='best')
            plt.savefig(f'Graphs/{dataset_name}/Experiment 2/{title}.png')
            # plt.show()
            plt.close()
            
            # plot TSNE
            kpca = KernelPCA(n_components=n_KPCA, kernel='rbf', random_state=1)
            transformed_X = kpca.fit_transform(X)
            
            # for p in range(10,210,10):
            #     plot_TSNE(transformed_X, title=f'TSNE (Perplexity {p}) for {dataset_name} Data\nReduced by {kernel}-KPCA to {n_KPCA} Components', p=p)
            
            sns.pairplot(pd.DataFrame(np.hstack((transformed_X,np.atleast_2d(y).T))), hue=n_KPCA, x_vars=[0], y_vars=[1])
            plt.savefig(f'Graphs/{dataset_name}/Experiment 2/Pairplot for KPCA with {n_KPCA}.png')
            plt.close()
            

def experiment_3(X, y, n_features, n_clusters_k_means, n_clusters_gmm, dataset_name):
    """Reproduce your clustering experiments, but on the data after you've run dimensionality 
    reduction on it. Yes, that’s 16 combinations of datasets, dimensionality reduction, and clustering 
    method. You should look at all of them, but focus on the more interesting findings in your report."""
    
    n_ICA = n_features['ICA']
    n_PCA = n_features['PCA']
    n_RP = n_features['RP']
    n_KPCA = n_features['KPCA']

    labels = []

    kernel = 'rbf'

    dimensionality_reducers = [(None, 'No'),
                               (FastICA(n_components=n_ICA, random_state=1), 'ICA'),
                               (PCA(n_components=n_PCA, random_state=1), 'PCA'),
                               (RP(n_components=n_RP, random_state=1), 'RP'),
                               (KernelPCA(n_components=n_KPCA, kernel=kernel, random_state=1), 'KPCA')]
    
    for reducer, reducer_name in dimensionality_reducers:
        print(f'\n\nExperiment 3 results for {reducer_name}')
        if reducer_name != 'No': X_transformed = reducer.fit_transform(X)
        else: X_transformed = X

        labels_k, labels_gmm = experiment_1(X=X_transformed, y=y, dataset_name=dataset_name, n_clusters_k_means=n_clusters_k_means, 
                                            n_clusters_gmm=n_clusters_gmm, dim_red=reducer_name, save_graphs=True)
        labels.append((reducer_name + ' k', labels_k))
        labels.append((reducer_name + ' gmm', labels_gmm))
        
        X_df = pd.DataFrame(X_transformed)
        with pd.ExcelWriter(f'{dataset_name}_{reducer_name}.xlsx') as writer:
                X_df.to_excel(writer, sheet_name=f'{reducer_name} X')
    
    labels.append(('true labels', y))

    print('Adjusted Rand:')
    for (name0, label0), (name1, label1) in itertools.combinations(labels, 2):
        print(f'{name0, name1}: {metrics.adjusted_rand_score(label0, label1)}')
    
    print('Adjusted Mutual Info:')
    for (name0, label0), (name1, label1) in itertools.combinations(labels, 2):
        print(f'{name0, name1}: {metrics.adjusted_mutual_info_score(label0, label1)}')

    print('Homogeneity:')
    for (name0, label0), (name1, label1) in itertools.combinations(labels, 2):
        print(f'{name0, name1}: {metrics.homogeneity_score(label0, label1)}')

    clusters_df = pd.DataFrame(np.array([l[1] for l in labels]).T, columns=[l[0] for l in labels])

    with pd.ExcelWriter(f'{dataset_name}_clusters.xlsx') as writer:
                clusters_df.to_excel(writer, sheet_name='Clusters')


def experiment_4(X_train, y_train, X_test, y_test, n_features, nn_params, dataset_name):
    """Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if 
    you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done 
    this) and rerun your neural network learner on the newly projected data."""
    
    n_ICA = n_features['ICA']
    n_PCA = n_features['PCA']
    n_RP = n_features['RP']
    n_KPCA = n_features['KPCA']

    kernel = 'rbf'

    dimensionality_reducers = [(None, 'No'),
                               (FastICA(n_components=n_ICA, random_state=1), 'ICA'),
                               (PCA(n_components=n_PCA, random_state=1), 'PCA'),
                               (RP(n_components=n_RP, random_state=1), 'RP'),
                               (KernelPCA(n_components=n_KPCA, kernel=kernel), 'KPCA')]
    
    for reducer, reducer_name in dimensionality_reducers:
        print(f'\nNeural Network Experiment 4 results for {reducer_name} Reduction')
        if reducer_name != 'No':
            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test)
        else:
            X_train_reduced = X_train
            X_test_reduced = X_test

        gs_params = {'hidden_layer_sizes': [(5), (10),(5,5),(5,10,5),(50),(50,50),(100),(200),(400)],
                     'learning_rate_init': [0.0025*n for n in range(1,11)]}

        print_gridsearch_results(clf=MLPClassifier(max_iter=10000), parameters=gs_params, X_train=X_train_reduced, y_train=y_train)

        hidden_layers, learning_rate = nn_params[reducer_name]
        predict_with_nn(X_train=X_train_reduced, y_train=y_train, X_test=X_test_reduced, y_test=y_test, 
                        hidden_layers=hidden_layers, learning_rate=learning_rate, dataset_name=dataset_name)


def experiment_5(X_train, y_train, X_test, y_test, n_clusters_k_means, n_clusters_gmm, nn_params, dataset_name):
    """Apply the clustering algorithms to the same dataset to which you just applied the dimensionality 
    reduction algorithms (you've probably already done this), treating the clusters as if they were new 
    features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. 
    Again, rerun your neural network learner on the newly projected data."""

    k_means_k = n_clusters_k_means['No']
    gmm_k = n_clusters_gmm['No']

    clusterers = [(KMeans(n_clusters=k_means_k, verbose=0), 'K-Means'), 
                  (GaussianMixture(n_components=gmm_k, random_state=1, verbose=0), 'GMM')]
    
    for clusterer, clusterer_name in clusterers:
        print(f'\nNeural Network Experiment 5 results for {clusterer_name}')
        clusterer.fit(X_train)
        encoder = OneHotEncoder()
        X_train_clustered = encoder.fit_transform(np.atleast_2d(clusterer.predict(X_train)).T)
        X_test_clustered = encoder.fit_transform(np.atleast_2d(clusterer.predict(X_test)).T)

        gs_params = {'hidden_layer_sizes': [(5), (10),(5,5),(5,10,5),(50),(50,50),(100),(200),(400)],
                     'learning_rate_init': [0.0025*n for n in range(1,11)]}

        print_gridsearch_results(clf=MLPClassifier(max_iter=10000), parameters=gs_params, X_train=X_train_clustered, y_train=y_train)

        hidden_layers, learning_rate = nn_params[clusterer_name]
        predict_with_nn(X_train=X_train_clustered, y_train=y_train, X_test=X_test_clustered, y_test=y_test, 
                        hidden_layers=hidden_layers, learning_rate=learning_rate, dataset_name=dataset_name)


def initial_visualization(data, X, y, column):
    visualizer = Rank2D(algorithm='pearson')
    visualizer.fit_transform(X)
    visualizer.show()


def print_gridsearch_results(clf, parameters, X_train, y_train, verbose=False):
    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, scoring=scorer)
    grid_search.fit(X_train, y_train)

    # source for this function: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    print('Best parameters set found on development set:')
    print()
    print(grid_search.best_params_)
    if verbose:
        print()
        print('Grid scores on development set:')
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print('%0.3f (+/-%0.03f) for %r'
                % (mean, std * 2, params))
    print()
    print(f'Best score for {str(clf)}: {grid_search.best_score_}')


def predict_with_nn(X_train, y_train, X_test, y_test, hidden_layers, learning_rate, dataset_name):
    if dataset_name == 'Telescope':
        beta = 0.5
    else:
        beta = 1

    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate_init=learning_rate, max_iter=1000, random_state=1)
    
    begin_train_time = perf_counter()
    clf.fit(X_train, y_train)
    end_train_time = perf_counter()

    train_time = end_train_time - begin_train_time

    begin_predict_time = perf_counter()
    train_predictions = np.squeeze(clf.predict(X_train))
    test_predictions = np.squeeze(clf.predict(X_test))
    end_predict_time = perf_counter()

    predict_time = end_predict_time - begin_predict_time
    
    train_score = fbeta_score(y_true=y_train, y_pred=train_predictions, beta=beta)
    test_score = fbeta_score(y_true=y_test, y_pred=test_predictions, beta=beta)

    print('\nGot: %.2f%% F-%s on the test set and %.2f%% F-%s on the train set for the model.' % 
                (test_score*100, str(beta), train_score*100, str(beta)))
    print(f'Model took {round(train_time,5)} sec to train and {round(predict_time,5)} sec to predict.')
    print(f'Model took {len(clf.loss_curve_)} iterations to train.\n')


def run():
    np.random.seed(0)

    # red wine 
    n_features_rw = {
        'ICA': 8,  # 8 by eyeballing. 4 or 5 by knees (max & avg, respectively).
        'PCA': 5,  # knees: 5 for sum eigenvalues, 4 for avg eigenvalues
        'RP': 7,  # knee at 7
        'KPCA': 6
    }

    n_clusters_k_means_rw = {
        'No': 7, # or 5 or 37 or 81 (but these break the NN experiment; not all clusters have assignments)
        'ICA': 7, # 6?
        'PCA': 7, # 4?
        'RP': 3,
        'KPCA': 5
    }

    n_clusters_gmm_rw = {
        'No': 7, # or 4 or 81 or 97 or 14
        'ICA': 5, # 3 or 4 or 5
        'PCA': 6, # or 7
        'RP': 5, # or 3
        'KPCA': 5
    }

    nn_params_rw = {
        'No':      ((300),   0.003),
        'ICA':     ((400),   0.015),
        'PCA':     ((50,50), 0.005),
        'RP':      ((50,50), 0.0225),
        'KPCA':    ((50,50), 0.015),
        'K-Means': ((5,10,5), 0.02),
        'GMM':     ((5,10,5), 0.025)
    }

    # telescope
    n_features_tel = {
        'ICA': 7,
        'PCA': 5, # 5 by sum, 3 by avg
        'RP': 9,
        'KPCA': 4 # 3 for polykernel by sum, 4 for RBF, 3 by avg for sigmoid, 5 by sum for sigmoid
    }

    n_clusters_k_means_tel = {
        'No': 3,
        'ICA': 8,
        'PCA': 3,
        'RP': 2, # or 5
        'KPCA': 5 # rbf kernel
    }

    n_clusters_gmm_tel = {
        'No': 4, # or 2 or 3
        'ICA': 4, # or 4 or 7
        'PCA': 4, # or 2
        'RP': 4, # or 2
        'KPCA': 6 # or 10 or 5
    }

    nn_params_tel = {
        'No':      ((150), 0.003), # ((50,50), 0.02)
        'ICA':     ((400), 0.0125),
        'PCA':     ((5), 0.0125),
        'RP':      ((50), 0.005),
        'KPCA':    ((50), 0.0225),
        'K-Means': ((5), 0.0025),
        'GMM':     ((5), 0.0025)
    }

    datasets = ['Red Wine Quality','Telescope']

    for dataset in datasets:
        print(f'\nResults for {dataset}:')
        
        if dataset == 'Red Wine Quality':
            data, X, y = import_red_wine_data()
            n_clusters_k_means = n_clusters_k_means_rw
            n_clusters_gmm = n_clusters_gmm_rw
            n_features = n_features_rw
            nn_params = nn_params_rw

        else:
            assert dataset == 'Telescope', 'Dataset import error in initial run step.'
            data, X, y = import_telescope_data()
            n_clusters_k_means = n_clusters_k_means_tel
            n_clusters_gmm = n_clusters_gmm_tel
            n_features = n_features_tel
            nn_params = nn_params_tel


        # initial_visualization(data, X, y, column='alcohol')
        X_train, X_test, y_train, y_test = split_and_scale(X, y)
        
        # clustering
        _, _ = experiment_1(X=X_train, y=y_train, n_clusters_k_means=n_clusters_k_means, n_clusters_gmm=n_clusters_gmm, 
                        dataset_name=dataset, save_graphs=True)
        
        # dimensionality reduction
        experiment_2(X=X_train, y=y_train, n_features=n_features, dataset_name=dataset, save_graphs=True)

        # dimensionality reduction then clustering
        experiment_3(X=X_train, y=y_train, n_features=n_features, n_clusters_k_means=n_clusters_k_means, 
                        n_clusters_gmm=n_clusters_gmm, dataset_name=dataset)

        # dimensionality reduction then nn
        experiment_4(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_features=n_features, 
                        nn_params=nn_params, dataset_name=dataset)
        
        # clustering then nn
        experiment_5(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_clusters_k_means=n_clusters_k_means, 
                        n_clusters_gmm=n_clusters_gmm, nn_params=nn_params, dataset_name=dataset)



run()