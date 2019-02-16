import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import random
import copy
import math

# seed RNG for reproducable results
random.seed(1)

# matplotlib colors for clusters
colors = ['r', 'b', 'g', 'c', 'm', 'y']

# percent change in centroid position to be
# considered significant, the higher the value,
# the faster and less accurate the algorithm
percent_change_minimum = 0.03

# returns the Euclidean distance between the two given points.
# parameters a and b are 1 by 2 arrays representing points on
# the Cartesian plane
def distance(a, b):
	return ((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2)


def gauss_prob_2d(gauss2d, point):
	x_mean = gauss2d[0][0]
	x_std_dev = gauss2d[0][1]

	y_mean = gauss2d[1][0]
	y_std_dev = gauss2d[1][1]

	x_portion = ((point[0] - x_mean) ** 2) / (2 * (x_std_dev ** 2))
	y_portion = ((point[1] - y_mean) ** 2) / (2 * (y_std_dev ** 2))

	return math.exp(-(x_portion + y_portion))


# produces random points within pandas DataFrame to use as
# initial centroid positions
def random_point(data):

	# get data range
	min_x = min(data[:,0])
	max_x = max(data[:,0])
	min_y = min(data[:,1])
	max_y = max(data[:,1])

	# pick random point from data range
	random_x = random.uniform(min_x, max_x)
	random_y = random.uniform(min_y, max_y)

	return [random_x, random_y]


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

# determines if any significant change was made in the
# position of the centroids
def centroids_were_updated(centroids, old_centroids):

	if old_centroids == []:
		return True

	for i in range(len(centroids)):
		x_change_percent = abs((centroids[i][0] - old_centroids[i][0]) / centroids[i][0])
		y_change_percent = abs((centroids[i][1] - old_centroids[i][1]) / centroids[i][1])

		if x_change_percent > percent_change_minimum or y_change_percent > percent_change_minimum:
			return True

	return False


# k-means clustering algorithm, generates matplotlib figure plotting clusters
# file: path to csv file containing data
# x: column name from csv file to plot on x-axis
# y: column name from csv file to plot on y-axis
# k: number of clusters to make (must be between 1 and 6 inclusive)
def kmeans(file_name, x, y, k):

	# check if given value of clusters is supportable
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

	# give centroids random initial positions
	for i in range(k):
		#centroids.append(random_point(data))
		centroids.append(random_gaussian(data))

	# construct a new matplotlib figure
	plt.figure()
	plt.xlabel(x)
	plt.ylabel(y)

	# loop until centroids stop moving
	# while centroids_were_updated(centroids, old_centroids):
	while centroids != old_centroids:

		# # stats used for updating centroids
		# x_totals = np.zeros(k)
		# y_totals = np.zeros(k)
		# counts = np.zeros(k)
		# distances = np.zeros(k)

		# store points values instead of sums to calculate variance
		x_values = [[] for i in range(k)]
		y_values = [[] for i in range(k)]
		probabilities = np.zeros(k)

		# iterate through every point in the scatter plot
		for i in data:

			# # find the nearest centroid to that point
			# for j in range(k):
			# 	distances[j] = distance(i, centroids[j])
			# closest_centroid = np.argmin(distances)

			# find probability that that point comes from each gaussian
			for j in range(k):
				probabilities[j] = gauss_prob_2d(centroids[j], i)
			most_likely_centroid = np.argmax(probabilities)


			# record stats about that point and make it the color
			# of its nearest centroid
			# x_totals[closest_centroid] += i[0]
			# y_totals[closest_centroid] += i[1]
			# counts[closest_centroid] += 1
			# plt.scatter(i[0], i[1], c=colors[closest_centroid])

			x_values[most_likely_centroid].append(i[0])
			y_values[most_likely_centroid].append(i[1])
			plt.scatter(i[0], i[1], c=colors[most_likely_centroid])

		# remember where the centroids used to be
		old_centroids = copy.deepcopy(centroids)

		# update centroid positions
		# for j in range(k):
		# 	if counts[j] != 0:
		# 		centroids[j] = [x_totals[j] / counts[j], y_totals[j] / counts[j]]

		# update centroid positions
		for j in range(k):
			if x_values[j] != 0:
				new_mean_x = np.mean(x_values[j])
				new_mean_y = np.mean(y_values[j])

				new_std_dev_x = np.std(x_values[j])
				new_std_dev_y = np.std(y_values[j])

				centroids[j] = [(new_mean_x, new_std_dev_x), (new_mean_y, new_std_dev_y)]
	
	# # plot centroids
	# for i in range(k):
	# 	plt.scatter(centroids[i][0], centroids[i][1], s=150, c=colors[i], marker='X', edgecolors='k')