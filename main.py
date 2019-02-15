import kmeans
import matplotlib.pyplot as plt

file_path = 'data/iris_flowers.csv'
x_axis = 'PetalLengthCm'
y_axis = 'SepalWidthCm'
kmeans.kmeans(file_path, x_axis, y_axis, 2)

file_path = 'data/auto-mpg.csv'
x_axis = 'weight'
y_axis = 'acceleration'
kmeans.kmeans(file_path, x_axis, y_axis, 3)

plt.show()