# PCA has been around for 100 years!

This is my implementation of principal component analysis as part of the final project for the Mathematics for Machine Learning: PCA specialization from Imperial College London[<a href="https://www.coursera.org/account/accomplishments/certificate/PBYVD7BRK88U" title="coursera.com" rel="nofollow">1</a></li>], hosted by Coursera. 

If we have some large group of data living in two-dimensions that we need to compress in order to find the most important patterns in that data for a given problem, we need to find a good one-dimensional subspace to minimize the average squared reconstruction error (loss function) of the data points. The ASRE being the variance of a ratio of summed squares equivalent to Pythagoras’s theorem, which defines the principal directions of PCA’s orthogonal basis vectors.

<p align="center">
  <img src="https://media.giphy.com/media/lly63TqgYYsoNWhjzc/giphy.gif">
</p>

When plotting the original dataset with their original projections onto a one-dimensional subspace, you can see that some   projections are more informative than others, and PCA will find the best projection with respect to its partial derivatives and their parameters using the multivariate chain rule by maximizing the variance of the data points after projection. This allows us to choose the number of principal components which by design allows us to remove variables from the data that are least important to correlation. 

The assignment required:

1. An implementation of the main steps of PCA
2. The application of PCA to an image dataset with a limited number of samples
3. An implementation of PCA for a high-dimensional version of the same dataset
4. The time/space complexity between both versions of the algorithm
