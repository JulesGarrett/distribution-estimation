import em
import matplotlib.pyplot as plt

# generate k-means cluster plot for 1st data set
file_path = 'data/iris_flowers.csv'
x_axis = 'PetalLengthCm'
y_axis = 'SepalWidthCm'
em.em(file_path, x_axis, y_axis, 2)

# generate k-means cluster plot for 2nd data set
file_path = 'data/winequality-red.csv'
x_axis = 'citric acid'
y_axis = 'volatile acidity'
em.em(file_path, x_axis, y_axis, 2)

# show both plots
plt.show()