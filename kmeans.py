import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

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


# k-means clustering algorithm
# file: path to csv file containing data
def kmeans(file_name, x, y, k):

	# check if given value of k is valid
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
		centroids.append(random_point(data))

	# construct a new matplotlib figure
	plt.figure()
	plt.xlabel(x)
	plt.ylabel(y)

	while centroids_were_updated(centroids, old_centroids):

		# stats used for updating centroids
		x_totals = np.zeros(k)
		y_totals = np.zeros(k)
		counts = np.zeros(k)
		distances = np.zeros(k)

		# iterate through every point in the scatter plot
		for i in data:

			# find the nearest centroid to that point
			for j in range(k):
				distances[j] = distance(i, centroids[j])
			closest_centroid = np.argmin(distances)

			# record stats about that point and make it the color
			# of its nearest centroid
			x_totals[closest_centroid] += i[0]
			y_totals[closest_centroid] += i[1]
			counts[closest_centroid] += 1
			plt.scatter(i[0], i[1], c=colors[closest_centroid])

		# remember where the centroids used to be
		old_centroids = copy.deepcopy(centroids)

		# update centroid positions
		for j in range(k):
			if counts[j] != 0:
				centroids[j] = [x_totals[j] / counts[j], y_totals[j] / counts[j]]
	
	# plot centroids
	for i in range(k):
		plt.scatter(centroids[i][0], centroids[i][1], s=150, c=colors[i], marker='X', edgecolors='k')