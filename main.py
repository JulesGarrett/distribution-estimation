import em
import matplotlib.pyplot as plt

print('Generating plot for first data set... \t', end='', flush=True)

# generate EM cluster plot for 1st data set
file_path = 'data/iris_flowers.csv'
x_axis = 'PetalLengthCm'
y_axis = 'SepalWidthCm'
n_clusters = 2
em.em(file_path, x_axis, y_axis, n_clusters)

print('first plot generated')
print('Generating plot for second data set... \t', end='', flush=True)

# generate EM cluster plot for 2nd data set
file_path = 'data/winequality-red.csv'
x_axis = 'citric acid'
y_axis = 'volatile acidity'
n_clusters = 2
em.em(file_path, x_axis, y_axis, n_clusters)

print('second plot generated')

# show both plots
plt.show()