import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




tf.reset_default_graph()
N = tf.placeholder('int64', name="input_to_your_function")
result = tf.reduce_sum(tf.range(N)**2)

sess = tf.Session()
print(result.eval({N: 10**5}, sess))
writer = tf.summary.FileWriter("logs/example", graph=sess.graph)

with tf.name_scope("Placeholders_examples"):
    arbitrary_input = tf.placeholder('float32')
    input_vector = tf.placeholder('float32', shape=(None,))
    fixed_vector = tf.placeholder('int32', shape=(10,))
    input_matrix = tf.placeholder('float32', shape=(None, 15))

    double_the_vector = input_vector*2
    elementwise_cosine = tf.cos(input_vector)
    vector_squares = input_vector**2 - input_vector + 1

my_vector = tf.placeholder('float32', shape=(None,), name="VECTOR_1")
my_vector2 = tf.placeholder('float32', shape=(None,))
my_transformation = my_vector * my_vector2 / (tf.sin(my_vector) + 1)
# print(my_transformation)
dummy = np.arange(5).astype('float32')
# print(dummy)
# print(dummy[::-1])
# print(my_transformation.eval({my_vector: dummy, my_vector2: dummy[::-1]}, sess))
writer.add_graph(my_transformation.graph)
writer.flush()

with tf.name_scope("MSE"):
    y_true = tf.placeholder("float32", shape=(None,), name="y_true")
    y_predicted = tf.placeholder("float32", shape=(None,), name="y_predicted")
    # Implement MSE(y_true, y_predicted), use tf.reduce_mean(...)
    # mse = ### YOUR CODE HERE ###
    mse = tf.reduce_mean(tf.square(y_predicted - y_true))

def compute_mse(vector1, vector2, s):
    return mse.eval({y_true: vector1, y_predicted: vector2}, s)

# print(compute_mse(dummy, dummy[::-1], sess))


shared_vector_1 = tf.Variable(initial_value=np.ones(5), name="example_variable")
sess.run(tf.global_variables_initializer())
# print("Initial value", sess.run(shared_vector_1))
sess.run(shared_vector_1.assign(np.arange(5)))
# print("New Value", sess.run(shared_vector_1))


my_scalar = tf.placeholder('float32')
scalar_squared = my_scalar**2
derivative = tf.gradients(scalar_squared, [my_scalar, ])

x = np.linspace(-3, 3)
x_squared, x_squared_der = sess.run([scalar_squared, derivative[0]], {my_scalar:x})
plt.plot(x, x_squared,label="$x^2$")
plt.plot(x, x_squared_der, label=r"$\frac{dx^2}{dx}$")
plt.legend()
plt.show()

