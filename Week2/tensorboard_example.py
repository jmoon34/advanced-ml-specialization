import tensorflow as tf
import numpy as np

tf.reset_default_graph()
x = tf.get_variable("x", shape=(), dtype=tf.float32)
f = x**2

optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)