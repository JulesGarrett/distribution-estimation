import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import random
import copy
import math

# seed RNG for reproducable results
random.seed(1)

# matplotlib colors for clusters
colors = ['r', 'b', 'g', 'c', 'm', 'y']


# calculates the probability that a given point belongs to a given 2d gaussian distribution
# gauss2d: parameters describing a 2d gaussian distribution [(x_mean, x_std_dev), (y_mean, y_std_dev)]
# point: 2d point on Cartesian plane (x, y)
# return: https://wikimedia.org/api/rest_v1/media/math/render/svg/32a8040ca1ef2b9e0ffa5664326ac4a150f91cea (where A = 1)
def gauss_prob_2d(gauss2d, point):
	x_mean = gauss2d[0][0]
	x_std_dev = gauss2d[0][1]

	y_mean = gauss2d[1][0]
	y_std_dev = gauss2d[1][1]

	x_portion = ((point[0] - x_mean) ** 2) / (2 * (x_std_dev ** 2))
	y_portion = ((point[1] - y_mean) ** 2) / (2 * (y_std_dev ** 2))

	# note: not normalized
	return math.exp(-(x_portion + y_portion))


# generates random gaussian parameters
# data: set of data points to run EM algorithm on
# return: 2d gaussian parameters in the format: [(x_mean, x_std_dev), (y_mean, y_std_dev)]
def random_gaussian(data):

	# get data range
	min_x = min(data[:,0])
	max_x = max(data[:,0])
	min_y = min(data[:,1])
	max_y = max(data[:,1])

	# pick random mean values
	random_mean_x = random.uniform(min_x, max_x)
	random_mean_y = random.uniform(min_y, max_y)

	# pick random standard deviation values
	random_std_dev_x = random.uniform(min_x, max_x) / 2
	random_std_dev_y = random.uniform(min_y, max_y) / 2

	# return random 2d gaussian stats
	return [(random_mean_x, random_std_dev_x), (random_mean_y, random_std_dev_y)]


# expectation-maximization clustering algorithm, generates matplotlib figure plotting clusters
# file_name: path to csv file containing data
# x: column name from csv file to plot on x-axis
# y: column name from csv file to plot on y-axis
# k: number of clusters to make (must be between 1 and 6 inclusive)
def em(file_name, x, y, k):

	# check if given value of clusters is supportable (based on number of colors we have declared)
	if k > 6 or k < 1:
		print('Error: number of clusters must be between 1 and 6 inclusive')
		print('Given number of clusters: ' + str(k))
		return

	# read csv, get desired column data
	df = pd.read_csv(file_name)
	data = np.column_stack((df[x], df[y]))

	# keep track of new and old centroid positions
	centroids = []
	old_centroids = []

	# give centroids random initial positions and variances
	for i in range(k):
		centroids.append(random_gaussian(data))

	# instantiate a new matplotlib figure
	plt.figure()
	a = plt.subplot()
	plt.xlabel(x)
	plt.ylabel(y)

	# loop until centroids stop changing
	while centroids != old_centroids:

		# keep statistics on points to update centroids
		x_values = [[] for i in range(k)]
		y_values = [[] for i in range(k)]
		probabilities = np.zeros(k)

		# iterate through every point
		for i in data:

			# find gaussian that each point most likely belongs to (also known as the E-step)
			for j in range(k):
				probabilities[j] = gauss_prob_2d(centroids[j], i)
			most_likely_centroid = np.argmax(probabilities)

			x_values[most_likely_centroid].append(i[0])
			y_values[most_likely_centroid].append(i[1])

		# remember where the centroids used to be
		old_centroids = copy.deepcopy(centroids)

		# update centroid positions (also known as the M-step)
		for i in range(k):
			if x_values[i] != 0:
				new_mean_x = np.mean(x_values[i])
				new_mean_y = np.mean(y_values[i])

				new_std_dev_x = np.std(x_values[i])
				new_std_dev_y = np.std(y_values[i])

				centroids[i] = [(new_mean_x, new_std_dev_x), (new_mean_y, new_std_dev_y)]
	
	# Here, the algorithm has determined the clusters, now we just need to plot the results

	# for each cluster
	for i in range(k):

		# get the points that belong to that cluster
		points = list(zip(x_values[i], y_values[i]))

		# and plot them
		for point in points:
			plt.scatter(point[0], point[1], c=colors[i])

		# get ellipse position
		x = centroids[i][0][0]
		y = centroids[i][1][0]

		# get ellipse size and angle, credit for these 6 lines goes to Jaime on stack overflow
		# (source: https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib)
		cov = np.cov(x_values[i], y_values[i])
		lambda_, v = np.linalg.eig(cov)
		lambda_ = np.sqrt(lambda_)
		width = lambda_[0] * 2 * (i + 1)
		height = lambda_[1] * 2 * (i + 1)
		angle = math.degrees(math.acos(v[0, 0]))

		# instantiate ellipse object
		e = Ellipse((x, y), width, height, angle)

		# add ellipse to figure
		e.set_clip_box(a.bbox)
		e.set_alpha(0.25)
		e.set_facecolor(colors[i])
		a.add_artist(e)