import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/data.csv')

x = 1

for i in range(2 * x, 2 * x + 2):
	for j in range(len(list(df))):
		if i == j: continue
		plt.figure()
		plt.scatter(df[(list(df)[i])], df[(list(df)[j])])
		plt.xlabel(list(df)[i])
		plt.ylabel(list(df)[j])


plt.show()