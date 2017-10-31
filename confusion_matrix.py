import matplotlib.pyplot as plt
import numpy as np

m = np.array([[1, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]])


plt.matshow(m)
plt.colorbar()
plt.show()
