import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

file_path = 'data/Iris.csv'
x_axis = 'PetalLengthCm'
y_axis = 'PetalWidthCm'

def distance(a, b):
	return ((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2)

def random_point(data):
	min_x = int(min(data[:,0]))
	max_x = int(max(data[:,0]))
	min_y = int(min(data[:,1]))
	max_y = int(max(data[:,1]))

	random_x = random.randrange(min_x, max_x)
	random_y = random.randrange(min_y, max_y)

	return [random_x, random_y]

# read csv file
df = pd.read_csv(file_path)

# combine desired columns into a single list of coordinates
data = np.column_stack((df[x_axis], df[y_axis]))

random.seed(1)

# declare arbitrary initial centroid locations

# first centroid = red
# second centroid = blue

centroids = [random_point(data),
			 random_point(data)]

# make sure old_centroids is initially different from centroids
old_centroids = [[centroids[0][0] - 1, centroids[0][1] - 1],
				 [centroids[1][0] - 1, centroids[1][1] - 1]]

while centroids != old_centroids:
	red_centroid_total_x = red_centroid_total_y = 0
	blue_centroid_total_x = blue_centroid_total_y = 0
	red_point_count = blue_point_count = 0

	for i in data:

		distance_to_red_centroid = distance(i, centroids[0])
		distance_to_blue_centroid = distance(i, centroids[1])

		# color code each point to match the nearest centroid
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
	
	# get pre-centroid average
	old_centroids = centroids

	if red_point_count == 0:
		red_point_count = 1

	if blue_point_count == 0:
		blue_point_count = 1

	centroids = [[red_centroid_total_x / red_point_count, red_centroid_total_y / red_point_count],
				 [blue_centroid_total_x / blue_point_count, blue_centroid_total_y / blue_point_count]]	

plt.scatter(centroids[0][0], centroids[0][1], s=150, c='r', marker='X', edgecolors='k')
plt.scatter(centroids[1][0], centroids[1][1], s=150, c='b', marker='X', edgecolors='k')
plt.xlabel(x_axis)
plt.ylabel(y_axis)

plt.show()