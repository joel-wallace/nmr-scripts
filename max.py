import numpy as np

data = np.loadtxt("test12_bl.txt")

print(np.shape(data))

print(np.where(data == np.max(data)))
