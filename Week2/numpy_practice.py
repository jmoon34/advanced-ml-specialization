import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# def sinus2d(x, y):
#     return np.sin(x) + np.sin(y)
#
# x = np.linspace(0, 2*np.pi, 100)
# y = np.linspace(0, 2*np.pi, 100)
#
# xx, yy = np.meshgrid(x, y)
# z = sinus2d(xx, yy)
# plt.imshow(z, origin='lower', interpolation='none')
# plt.show()

patch = np.array([[1, 1],
                  [0, 1]])
kernel = np.array([[1, 2],
                   [3, 4]])

print(patch.dot(kernel))