# Dependencies
import timeit

import numpy as np
import pandas as pd

from load_data import load_mnist

from ipywidgets import interact

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline


# Load MNIST dataset
MNIST = load_mnist()
images, labels = MNIST['data'], MNIST['target']


# Quick plot of an image from the dataset to make sure data loaded properly
plt.figure(figsize = (4,4))
plt.imshow(images[0].reshape(28,28), cmap = 'gray');


'''
The next four functions preprocess the data so the images can have a zero mean and 
variance of one. The pixel encodings are originally uint8 and must be converted to 
a floating point number between 0-1. The mean mu, μ, must be subtracted from each image.
Each image must be scaled dimensionally by 1/σ, where σ is the standard deviation,
X_normalized will be the normalized dataset.
'''

def normalize(X):
    '''
    Normalize the dataset X.

    Parameters:
    X: ndarray of shape (N, D)

    Returns:
    X_normalized: ndarray of shape (N, D) - normalized dataset
    mean: ndarray of shape (D,) - mean of each feature
    std_dev: ndarray of shape (D,) - standard deviation of each feature
    
     In the future, you could encounter dimensions where the standard deviation is
     zero, for those when you do normalization the normalized data will be NaN. 
     Handle this by setting using `std = 1` for those dimensions when doing normalization.
    '''
    # Calculate the mean of each feature
    mu = np.mean(X, axis = 0) 
    # Calculate standard deviation of each feature
    std_dev = np.std(X, axis = 0)
    # Create copy of std_dev
    std_dev_filled = std_dev.copy()
    # Avoid division by zero by setting std_dev to 1 where it is 0
    std_dev_filled[std == 0] = 1.0
    # Normalize the dataset
    X_normalized = (X - mu) / std_dev_filled                  
    return X_normalized, mu, std_dev


def compute_eigen(S):
    ''' 
    Parameters:
    S: ndarray of shape (D, D)

    Returns:
    eigen_values: ndarray of shape (D,) - sorted eigenavlues
    eigen_vectors: ndarray of shape (D, D) - corresponding eigenvectors
     
    Computes the eigenvalues and corresponding eigenvectors for the covariance matrix S 
    and sorts them in descending order of eigenvalues.
    '''
    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(S)
    # Get indices of sorted eigenvalues in descending order
    sorted_indices = eigen_values.argsort()[:: - 1]
    # Sort eigenvalues
    eigen_values = eigen_values[sorted_indices]
    # Sort eigenvectors
    eigen_vectors = eigen_vectors[:, sorted_indices]
    return (eigen_values, eigen_vectors) 


def projection_matrix(B):
   '''
   Parameters:
   B = ndarray of shape (D, M) is the basis for the subspace 
   
   Returns:
   P = projection matrix of shape (D, D)
   
   Computes the projection matrix onto the space spanned by B.
   '''
   # Compute projection matrix
    return B @ (np.linalg.inv(B.T @ B)) @ B.T
 

def pca(X, num_components):
   '''
   Parameters:
   X: ndarray of size (N, D), where N is the number of datapoints and D is the 
   dimension of the data.
   
   num_components: int - number of principal components to use

   Returns:
   X_reconstructed: ndarray of shape (N, D) the reconstruction of X from the 
   first `num_components` principal components

   Perform principal component analysis (PCA) on dataset X.
   '''
    # Normalize the dataset
    X_normalized, mu, std_dev = normalize(X)
    # Compute covariance matrix
    eigen_values, eigen_vectors = np.cov(X_normalized, rowvar = False, bias = True)
    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = compute_eigens(covariance_matrix)
    # Compute projection matrix
    projection_matrix = compute_projection_matrix(eigen_vectors[:, :num_components])
    # Reconstructed dataset
    X_reconstructed = (projection_matrix @ X_normalized.T).T
    return X_reconstruct 


 def mse(predicted, actual):
    '''
    Parameters:
       predicted: ndarray of shape (N, D)
       actual: ndarray of shape (N, D)

    Returns: mse: float - mean squared error
    
    Computes the mean squared error (MSE) between predicted and actual data. In other
    words: how many principal components are needed in order to reach a mean squared 
    error of less than 100 for the dataset.
    '''
    # Calculate and return MSE
    return np.square(predicted - actual).sum(axis = 1).mean()


def pca_high_dimensional(X, num_components):
   '''
   Parameters:
   X: ndarray of size (N, D), where D is the dimension of the sample, N is 
   the number of samples.
   
   num_components: int - number of principal components to use
   
   Returns:
   X: ndarray of shape (N, D) the reconstruction of X from the first `num_components` 
   pricipal components.

   Performs PCA for high-dimensional datasets where the number of features is larger 
   than the number of samples (small sample size). When the dimensionality of the dataset 
   is larger than the number of given samples, PCA can be implemented in a more optimized 
   way for high-dimensionality.
   '''
   # Get dimensions of the dataset
   N, D = X.shape
   # Compute Gram matrix
   gram_matrix = (X @ X.T) / N
   # Compute eigenvalues and eigenvectors
   eigen_values, eigen_vectors = compute_eigen(gram_matrix)
   # Compute principle components
   principal_components = (X.T @ eigen_vectors)[:, :num_components]
   # Compute projection matrix
   projection_matrix = compute_projection_matrix(principal_components)
   # Reconstruct dataset
   X_reconstructed = (projection_matrix @ X.T).T
   return X_reconstruct 


def time(func, repeat=10):
   '''
   Parameters:
      func: function - function to measure time
      repeat: int - number of repetitions

   Return:
   mean_time: float - mean execution time
   std_time: float - standard deviation of execution time

   Measure the execution time of a function. Finds time/space complexity by 
   comparing the running time between pca and pca_high_dimensional functions.
   '''
   # Initialize list to store execution times
   times = []
   # Repeat measurement
   for _ in range(repeat):
      # Start timer
      start = timeit.default_timer()
      # Execute function
      func()
      # Stop timer
      stop = timeit.default_timer()
      # Append elaspsed time to list
      times.append(stop - start)
   # Return mean and standard deviation
   return np.mean(times), np.std(times)


# Data preprocessing
# Define number of datapoints to use
NUM_DATAPOINTS = 1000
# Reshape and normalize images
X = (images.reshape(-1, 28 * 28)[:NUM_DATAPOINTS]) / 255.0
print(X.shape)
# Normalize X
X_normalized, mu, std_dev = normalize(X)
print(mu.shape)
print(std_de.shape)


# The number of principal components we use, the smaller the reconstruction error will be
# Iterate PCA from sklearn
for num_component in range(1, 20):
    # Compare custom PCA implementation with sklean's implementation
    from sklearn.decomposition import PCA as SKPCA
    # Initialize sklearn PCA
    pca = SKPCA(n_components = num_component, svd_solver = 'full')
    # Reconstruct using sklearn PCA
    sklearn_reconstructed = pca.inverse_transform(pca.fit_transform(X_normalized))
    # Reconstruct using custom PCA
    custom_reconstructed = pca(X_normalized, num_component)
    # Assert almost equal
    np.testing.assert_almost_equal(custom_reconstructed, sklearn_reconstructed)
    print(np.square(reconst - sklearn_reconst).sum())

# Evaluate reconstruction error for different number of principal components
# Initialize list to store errors
reconstruction_errors = []
# Initialize list to store reconstructions
reconstructions = []


# Iterates over number of principal components
for num_component in range(1, 100):
    # Reconstruct using custom PCA
    reconstructed = pca(X_normalized, num_component)
    # Compute reconstruction error
    error = mean_squared_error(reconstructed, X_normalized)
    # Append reconstructed data to list
    reconstructions.append(reconstructed)
    # Append error to list
    loss.append((num_component, error))

# Convert reconstructions to ndarray
reconstructions = np.asarray(reconstructions)
# Unnormalize the reconstructed images
reconstructions = reconstructions * std_dev + mu
# Convert errors to ndarray
reconstruction_errors = np.asarray(reconstruction_errors)

# Creates a table showing the number of principal components and MSE
pd.DataFrame(reconstruction_errors).head(10)

# Plot MSE vs number of principal components
fig, ax = plt.subplots()
# Plot errors
ax.plot(reconstruction_errors[:,0], reconstruction_errors[:,1])
# Add horizontal line at MSE=100
ax.axhline(100, linestyle = '--', color = 'r', linewidth=2)
ax.xaxis.set_ticks(np.arange(1, 100, 5));
ax.set(xlabel =' num_components', ylabel = 'MSE', title='MSE vs Number of Principal Components');

# Validate the PCA_high_dim implementation
 np.testing.assert_almost_equal(pca(X_normalized, 2), pca_high_dimensional(X_normalized, 2))
    
# Variables to store measurements
# Initialize list to store times for X^TX and XX^T
times_xtx = []
times_xxt = []

# Iterate over datasets of different size by computing the running time of X^TX and XX^T
for dataset_size in np.arange(4, 784, step = 100):
    # Get subset of data
    subset_X = Xnormalized[:dataset_size]
    '''
    Set number of principal components, measure time for PCA,
    record the running time for computing X.T @ X'''
    mu_time, std_time = measure_time(lambda: pca(subset_X, num_principal_components), repeat=10)
    # Append result to list
    times_pca.append((dataset_size, mu_time, std_time))

# Convert times_pca to ndarray
times_pca = np.asarray(times_pca)
# Convert times_pca_hd to ndarray
times_pca_hd = np.asarray(times_pca_hd)

# Variables to store measurements
# Initialize list to store times for X^TX and XX^T
times_xtx = []
times_xxt = []

# Iterate over datasets of different size by computing the running time of X^TX and XX^T
for dataset_size in np.arange(4, 784, step = 100):
    # Get subset of data
    subset_X = X_normalized[:dataset_size]
    '''
    Set number of principal components, measure time for PCA,
    record the running time for computing X.T @ X'''
    mu_time, std_time = measure_time(lambda: subset_X.T @ subset_X)
    # Append result to list
    times_xtx.append((dataset_size, mu_time, std_time))

# Convert times_xtx to ndarray
times_xtx = np.asarray(times_xtx)
# Convert times_xxt to ndarray
times_xxt = np.asarray(times_xxt)

# Plots the running time of computing X @ X.T(X^TX) and X @ X.T(XX^T)
# Create figure and axes
fig, ax = plt.subplots()
# Set axis labels
ax.set(xlabel = 'size of dataset', ylabel = 'running time')
# Plot times for X^TX
bar = ax.errorbar(times_xtx[:, 0], times_xtx[:, 1], times_xtx[:, 2], label = '$X^T X$ (PCA)', linewidth = 2)
# Plot times for XX^T
ax.errorbar(times_xxt[:, 0], times_xxt[:, 1], times_xxt[:, 2], label = '$X X^T$ (PCA_high_dim)', linewidth = 2)
ax.legend()

# Measure execution time for PCA and PCA_high_dim
%time pca(X_normalized, 2)
%time pca_high_dimensional(X_normalized, 2)
pass # Put this here so that our output does not show result of computing `X_normalized @ X_normalized.T`

# Measure running time of PCA and PCA_high_dim for different dataset sizes
# Initialize list to store times for PCA
times_pca = []
# Initialize list to tstore times for PCA_high_dim
times_pca_hd = []

# Iterate over dataset sizes
for dataset_size in np.arange(4, 784, step = 100):
    # Get subset of data
    subset_X = X_normalized[:dataset_size]
    # Set number of principal components
    num_principal_components = 2
    # Measure time for PCA
    mu_time, std_time = measure_time(lambda: pca(subset_X, num_principal_components), repeat = 10)
    # Append result to list
    times_pca.append((dataset_size, mu_time, std_time))

    # Measure time for PCA_high_dim
    mu_time, std_time = measure_time(lambda: pca_high_dimensional(subset_X, num_principal_components), repeat = 10)
    times_pca_hd.append((dataset_size, mu_time, std_time))

# Convert times_pca to ndarray
times_pca = np.asarray(times_pca)
times_pca_hd = np.asarray(times_pca_hd)

# Plots the time/space complexity of each algorithm
# Create a new figure and axes
fig, ax = plt.subplots()
# Set axis labels
ax.set(xlabel = 'Number of Datapoints', ylabel = 'Run Time')
# Plot times for PCA
ax.errorbar(times_pca[:, 0], times_pca[:, 1], times_pca[:, 2], label = 'PCA', linewidth=2)
# Plot times for PCA_high_dim
ax.errorbar(times_pca_hd[:, 0], times_pca_hd[:, 1], times_pca_hd[:, 2], label = 'PCA_high_dim', linewidth=2)
ax.legend();

# Measure execution time for PCA and PCA_high_dim with 2 principal components
# Measure time for PCA with 2 components
%time pca(X_normalized, 2)
# Measure time for PCA_high_dim with 2 components
%time PCA_high_dim(Xbar, 2)
# Prevent output from showing result of computing 'X_normalized @ X_normalized.T'
pass
