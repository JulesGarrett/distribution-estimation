## EECS 738 Project 1: Distribution Estimation by Matthew Taylor

### Overview
The purpose of this project is to model data using mixture models of of probability distributions programmatically. This is accomplished by using the expectation-maximization clustering algorithm, which can categorize points into multivariate Gaussian distributions and can help illustrate these distributions graphically.

### Approach
Because the expectation-maximization works best on continuous, clustered data (like the famous Old Faithful Geyser Data set), and not at all on discrete data, I began by finding two data sets that offered compatible data. The most interesting data set I found in the UCI machine learning repository was the Iris Species data set. Not only did this data set contain a small number of continuous observations, it was highly clustered, making it perfect for the expectation-maximization algorithm.
