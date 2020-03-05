import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


N = 1000
D = 3
x = np.random.random((N, D))
w = np.random.random((D, 1))
y = x @ w + np.random.randn(N,1)*0.20

# placeholders for input data:
tf.reset_default_graph()
features = tf.placeholder(tf.float32, shape=(None,D))
target = tf.placeholder(tf.float32, shape=(None,1))

# make predictions:
weights = tf.get_variable("w", shape=(D,1), dtype=tf.float32)
predictions = features @ weights

# define loss
loss = tf.reduce_mean((target-predictions)**2)
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)

# solving linear regression
# Gradient descent:
s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())
for i in range(300):
    _, curr_loss, curr_weights = s.run(
        [step,loss,weights], feed_dict={features: x, target: y}
    )
    if i % 50 == 0:
        print(curr_loss)

print(curr_weights)