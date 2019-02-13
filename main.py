import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# seed RNG for reproducable results
random.seed(1)

# get csv files and desired columns
file_path = 'data/iris_flowers.csv'
x_axis = 'PetalLengthCm'
y_axis = 'SepalWidthCm'

# file_path = 'data/breast_cancer.csv'
# x_axis = 'concave points_worst'
# y_axis = 'texture_mean'


# returns the Euclidean distance between the two given points
def distance(a, b):
	return ((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2)

# produces random points within pandas DataFrame to use as
# initial centroid positions
def random_point(data):
	min_x = min(data[:,0])
	max_x = max(data[:,0])
	min_y = min(data[:,1])
	max_y = max(data[:,1])

	random_x = random.uniform(min_x, max_x)
	random_y = random.uniform(min_y, max_y)

	return [random_x, random_y]


# read csv, get desired column data
df = pd.read_csv(file_path)
data = np.column_stack((df[x_axis], df[y_axis]))

# declare initial centroid positions
centroids = [random_point(data),	# red centroid
			 random_point(data)]	# blue centroid

# make a second set of centroids so we can check if
# the centroid positions have changed, in order to stop
# the algorithm. it is important that we make sure
# old_centroids is initially different from centroids
old_centroids = [[centroids[0][0] - 1, centroids[0][1] - 1],
				 [centroids[1][0] - 1, centroids[1][1] - 1]]


# begin k-means clustering algorithm, repeat this until
# centroids stop moving
while centroids != old_centroids:

	# keep various records. these will be useful in finding
	# the position of the new centroids later
	red_centroid_total_x = red_centroid_total_y = 0
	blue_centroid_total_x = blue_centroid_total_y = 0
	red_point_count = blue_point_count = 0


	# iterate through every point in the scatter plot
	for i in data:

		# find the distance between each point and both of the centroids
		distance_to_red_centroid = distance(i, centroids[0])
		distance_to_blue_centroid = distance(i, centroids[1])

		# make each point the same color as its nearest centroid
		if distance_to_red_centroid < distance_to_blue_centroid:
			red_centroid_total_x += i[0]
			red_centroid_total_y += i[1]
			red_point_count += 1
			plt.scatter(i[0], i[1], c='r')
		else:
			blue_centroid_total_x += i[0]
			blue_centroid_total_y += i[1]
			blue_point_count += 1
			plt.scatter(i[0], i[1], c='b')
	

	# remember where the centroids used to be
	old_centroids = centroids

	# in the off chance that all of the points are one color,
	# change the other point count from zero to one to avoid
	# division by zero
	if red_point_count == 0:
		red_point_count = 1
	if blue_point_count == 0:
		blue_point_count = 1

	# update centroid poisitions to the average position of all of the points in its cluster
	centroids = [[red_centroid_total_x / red_point_count, red_centroid_total_y / red_point_count],
				 [blue_centroid_total_x / blue_point_count, blue_centroid_total_y / blue_point_count]]	

# end k-means clustering algorithm


# plot results
plt.scatter(centroids[0][0], centroids[0][1], s=150, c='r', marker='X', edgecolors='k')
plt.scatter(centroids[1][0], centroids[1][1], s=150, c='b', marker='X', edgecolors='k')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.show()