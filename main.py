import kmeans
import matplotlib.pyplot as plt

# generate k-means cluster plot for 1st data set
file_path = 'data/iris_flowers.csv'
x_axis = 'PetalLengthCm'
y_axis = 'SepalWidthCm'
kmeans.kmeans(file_path, x_axis, y_axis, 2)

# generate k-means cluster plot for 2nd data set
file_path = 'data/auto-mpg.csv'
x_axis = 'weight'
y_axis = 'acceleration'
kmeans.kmeans(file_path, x_axis, y_axis, 3)

# show both plots
plt.show()