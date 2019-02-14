import kmeans
import matplotlib.pyplot as plt

file_path = 'data/iris_flowers.csv'
x_axis = 'PetalLengthCm'
y_axis = 'SepalWidthCm'
kmeans.kmeans(file_path, x_axis, y_axis, 3)

file_path = 'data/breast_cancer.csv'
x_axis = 'radius_mean'
y_axis = 'texture_mean'
kmeans.kmeans(file_path, x_axis, y_axis, 3)

plt.show()