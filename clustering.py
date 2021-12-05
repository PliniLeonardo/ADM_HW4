import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
import seaborn as sns




def custom_kmeans(X, K, epsilon = 1/1000, maxIter = None):
    '''
    this function runs the kmeans algorithm on the dataset X
    searching for K clusters (labelled from  to K-1)

    Input: dataframe (for which the columns are the numerical features
            and the rows are the actual data points),
           number of clusters, parameter to control the convergence,
           maximum number of iterations to perform (if None, there will be no cap)

    Output: number of performed iterations, array of labels,
        centroids coordinates dataFrame, total cost of the solution
    '''
    
    should_Not_Stop = cycle_exit(epsilon, maxIter)
    
    npoints = len(X)
    
    ## random initialization
    # select a subset of points
    selected_points = np.random.choice(npoints, size=K, replace=False)
    # extract the centroids
    centroids = X.loc[selected_points].copy().reset_index(drop = True)
    
    # assign every point to the correct cluster
    labels = assign_cluster(X, centroids)
    
    # computing the cost of the initial solution
    cost = compute_cost(X,centroids, labels)
    
    
    # value to start the cycle
    previous_cost = cost + 1 + epsilon
    
    n_iter = 0
    while ( should_Not_Stop(cost, previous_cost, n_iter) ):
        n_iter += 1
        
        # update the centroids
        update_centroids(X,centroids, labels)
        
        # assign every point to the correct cluster
        labels = assign_cluster(X,centroids)
        
        # compute new cost
        previous_cost = cost
        cost = compute_cost(X,centroids, labels)

    return(n_iter, labels, centroids, cost)


def cycle_exit(epsilon, maxIter):
    '''
    function used to retrieve the correct function to be
    used for cycle exit evaluation
    '''

    def epsilon_function(cost, previous_cost, n_iter):
        exit = previous_cost - cost > epsilon
        return(exit)
    
    def iter_function(cost, previous_cost, n_iter):
        convergence = previous_cost - cost > epsilon
        iteration = n_iter <= maxIter
        return(convergence and iteration)


    if maxIter is not None:
        return (iter_function)
    else:
        return (epsilon_function)


def assign_cluster(X, centroids):
    '''
    this function assigns every data point in X
    and return the labels array
    '''
    K = len(centroids)
    npoints = len(X)
    nfeatures = X.shape[1]
    
    # we will use numpy methods to allow for faster computations
    # we will broadcast both X and the centroids to a 3D array of dimension (npoints, nfeatures, K)
    # in which the third dimension will represent the clusters
    broadcasting_shape = (npoints, nfeatures, K)
    
    # add the third dimension to the dataset and broadcast it
    broadcasted_X = X.values[:,:,np.newaxis]                                               # now the shape is (npoints, nfeatures, 1)
    broadcasted_X = np.broadcast_to(broadcasted_X, broadcasting_shape)                     # now the shape is (npoints, nfeatures, K)
    
    # transpose the centroids matrix and add a new dimension
    broadcasted_centroids = np.transpose(centroids.values)                                 # now the shape is          (nfeatures, K)
    broadcasted_centroids = broadcasted_centroids[np.newaxis,:,:]                          # now the shape is       (1, nfeatures, K)
    broadcasted_centroids = np.broadcast_to(broadcasted_centroids, broadcasting_shape)     # now the shape is (npoints, nfeatures, K)
    
    # here we compute the squared distance between every point and the centroid
    # the position (i,j) is the squared distance of the i-th point from the j-th centroid
    squared_distances = np.sum( (broadcasted_X - broadcasted_centroids)**2, axis=1)        # now the shape is (npoints, K)
    
    # here we take the argument which corresponds to the minimum distance
    labels = np.argmin(squared_distances, axis = 1)                                        # now the shape is  (npoints, )
    
    return (labels)

def update_centroids(X, centroids, labels):
    '''
    this function updates the centroids coordinates
    '''
    K = len(centroids)
    
    for i in range(K):
        centroids.loc[i] = X[labels == i].mean()
    
    return

def compute_cost(X, centroids, labels):
    '''
    computes the cost of a given solution
    '''
    
    K = len(centroids)
    
    cost = 0
    for i in range(K):
        # dataframe of squared differences
        squared_difference_vector = (X[labels == i] - centroids.loc[i])**2
        
        # we just add up every element of the matrix, since there is no square root to perform
        cost += np.sum(squared_difference_vector.values)

    return(cost)

def compute_silhouette(X, K, labels):
    '''
    computes the Silhouette score of a given solution
    
    input: dataset, number of clusters, labels
    '''
    
    npoints = len(X)
    
    # count the size of each cluster
    cluster_sizes = Counter(labels)
    
    array_score = []
    for i in range(npoints):
        # compute silhouette score of i-th data point
        array_score.append(silhouette_score(i, X, K, labels, cluster_sizes))
    
    # final silhouette score
    silhouette = np.array(array_score).mean()
    
    return(silhouette)
    
def silhouette_score(i, X, K, labels, cluster_sizes):
    '''
    computes the silhouette score for the i-th data point
    '''
    # cluster of che current data point
    current_cluster = labels[i]
    
    # get every other cluster
    every_other_cluster = list(range(K))
    every_other_cluster.remove(current_cluster)
    
    if cluster_sizes[current_cluster] == 1:
        return(0)
    
    a = silhouette_a(X.loc[i], X[labels == current_cluster], cluster_sizes[current_cluster])
    b = silhouette_b(X.loc[i], X[labels != current_cluster], labels[labels!= current_cluster], cluster_sizes, every_other_cluster)
    
    score = (b-a)/max(a,b)
    
    return(score)

def silhouette_a(data_point, X, size):
    '''
    computes the silhouette 'a' coefficient
    '''
    # compute distance between each point in the cluster and the current data point
    a_score = np.sum(np.linalg.norm( X-data_point , axis = 1))/(size-1)
    
    return(a_score)

def silhouette_b(data_point, X, labels, cluster_sizes, every_other_cluster):
    '''
    computes the silhouette 'b' coefficient
    '''
    #edge case
    if len(every_other_cluster) == 0:
        return(0)
    
    # first cluster
    cluster_idx = every_other_cluster[0]
    size = cluster_sizes[cluster_idx]
    temp_b_score = np.sum(np.linalg.norm( X[labels == cluster_idx] - data_point, axis = 1))/size
    
    final_b_score = temp_b_score
    
    for cluster_idx in every_other_cluster[1:]:
        size = cluster_sizes[cluster_idx]
        temp_b_score = np.sum(np.linalg.norm( X[labels == cluster_idx] - data_point, axis = 1))/size
        if temp_b_score < final_b_score:
            final_b_score = temp_b_score
    
    return(final_b_score)

def kmeans_plus_plus(X, K):
    '''
    this function runs the kmeans++ algorithm from sklearn on the dataset X
    searching for K clusters (labelled from  to K-1)

    Input: dataframe (for which the columns are the numerical features
            and the rows are the actual data points),
           number of clusters

    Output: number of performed iterations, array of labels,
        centroids coordinates dataFrame, total cost of the solution
    '''
    
    kmeans = KMeans(n_clusters=K)
    labels = kmeans.fit_predict(X)
    centroids = pd.DataFrame(kmeans.cluster_centers_)
    n_iter = kmeans.n_iter_
    cost = compute_cost(X, centroids, labels)
    
    return(n_iter, labels, centroids, cost)

def run_elbow_silhouette(X, K = 10):
    '''
    this function computes the cost and the silhouette coefficient for various
    cluster sizes and returns them as two arrays
    
    input: dataset, maximum number of clusters
    '''
    
    # initialize output array
    cost_array = []
    silhouette_array = []
    
    for i in tqdm(range(2,K)):
        n_iter, labels, centroids, cost = custom_kmeans(X, i)
        cost_array.append(cost)
        silhouette_array.append(compute_silhouette(X, i, labels))
    
    return(cost_array, silhouette_array)

def features_3dplot_comparison(X, K, custom_labels, plus_plus_labels):
    '''
    plots the 3d plots of the clustered data
    
    input: dataset, number of clusters, labels from kmeans, labels from kmeans++
    '''
    
    fig, ax = plt.subplots(1,2,figsize = (18,10), subplot_kw=dict(projection="3d"))

    for i in range(K):
        custom_x = X[custom_labels == i]['audio_feature_0']
        custom_y = X[custom_labels == i]['audio_feature_1']
        custom_z = X[custom_labels == i]['audio_feature_2']

        plus_plus_x = X[plus_plus_labels == i]['audio_feature_0']
        plus_plus_y = X[plus_plus_labels == i]['audio_feature_1']
        plus_plus_z = X[plus_plus_labels == i]['audio_feature_2']

        ax[0].scatter(custom_x, custom_y, custom_z, s = 20)
        ax[1].scatter(plus_plus_x, plus_plus_y, plus_plus_z, s = 20)

    ax[0].set_title('KMeans')
    ax[1].set_title('KMeans++')

    plt.show()
    
    return