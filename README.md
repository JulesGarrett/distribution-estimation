## EECS 738 Project 1: Distribution Estimation by Matthew Taylor

### Overview
The purpose of this project is to model data using mixture models of of probability distributions programmatically. This is accomplished by using the expectation-maximization clustering algorithm, which can categorize points into multivariate Gaussian distributions and can help illustrate these distributions graphically.

### Approach
Because the expectation-maximization works best on continuous, clustered data (like the famous Old Faithful Geyser data set), and not well at all on discrete data, I began by finding two data sets that offered compatible data. The most interesting data set I found in the UCI machine learning repository was the iris species data set. Not only did this data set contain a small number of continuous observations, it was highly clustered, making it perfect for the expectation-maximization algorithm. The other data set I decided to use for this project was the red wine quality data set, which has many of the features that the iris data set has.

Once the data sets were chosen, I began by implementing the K-Means clustering algorithm. I thought this was the most logical first step as expectation-maximization is simply an extension of K-Means that clusters based on probability distributions rather than Euclidean distance. Once K-Means had been implemented, I swapped out the basic Euclidean distance function with one that gave me the probability that a point belonged to a given multivariate Gaussian distribution, and I was essentially finished. The only thing left to do was plot the results and my estimations of the probability distributions using matplotlib.

### How To Run
This project was written in Python 3.7.2 and relies on matplotlib, numpy, and pandas. To install these packages, run this command: `pip3 install matplotlib numpy pandas`. Once the packages are installed, simply navigate to the project directory in a command line and enter:  `python3 main.py` on Linux, or `py main.py` on Windows. Everything is already setup inside main.py so the user doesn't need to pass in any parameters.

### Results
The program takes about 5 seconds to execute and displays two figures, one for each data set. Figure 1 contains data from the iris species data set and Figure 2 contains data from the red wine quality data set. Both figures are shown below.

### Figure 1: Iris Species
![Figure 1](https://i.imgur.com/3treGtF.png)

### Figure 2: Red Wine Quality
![Figure 2](https://i.imgur.com/l3p9Lcu.png)
