import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from Week2 import preprocessed_mnist
X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_mnist.load_dataset()

def visuazlize_loss_accuracy(t_loss, v_loss, t_acc, v_acc):
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.set_title("Loss")
    ax1.plot(train_losses, c='r', label='train_loss')
    ax1.plot(val_losses, c='b', label='val_loss')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(122)
    ax2.set_title("Accuracy")
    ax2.plot(train_accuracies, c='r', label='train_accuracy')
    ax2.plot(valid_accuracies, c='b', label='valid_accuracy')
    ax2.legend(loc="lower right")
    return fig


print("X_train [shape {s}] sample patch:\n".format(s=str(X_train.shape)), X_train[1, 15:20, 5:10])
print("A closeup of a sample patch:")
plt.imshow(X_train[1, 15:20, 5:10], cmap="Greys")
plt.show()
print("And the whole sample:")
plt.imshow(X_train[1], cmap="Greys")
plt.show()
print("y_train [shape {s}] 10 samples:\n".format(s=str(y_train.shape)), y_train[:10])

X_train_flat = X_train.reshape((X_train.shape[0], -1))
# print(X_train_flat.shape)
X_val_flat = X_val.reshape((X_val.shape[0], -1))
# print(X_val_flat.shape)

y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)

s = tf.Session()

# Model parameters: W and b
W = tf.get_variable('W', shape=(784, 10))
b = tf.get_variable('b', shape=(10,), initializer=tf.zeros_initializer())

# Placeholders for the input data
input_X = tf.placeholder('float32', shape=(None, 784), name='input_X')
input_y = tf.placeholder('float32', shape=(None, 10), name='input_y')

# Compute predictions
logits = input_X @ W + b
probs = tf.nn.softmax(logits)
classes = tf.argmax(probs, axis=1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits))
step = tf.train.AdamOptimizer().minimize(loss)

# Run and graph
s.run(tf.global_variables_initializer())
BATCH_SIZE = 512
EPOCHS = 40

train_losses = []
val_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(EPOCHS):
    batch_losses = []
    for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE):
        _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start+BATCH_SIZE],
                                input_y: y_train_oh[batch_start:batch_start+BATCH_SIZE]})
        batch_losses.append(batch_loss)
    train_losses.append(np.mean(batch_losses))
    val_losses.append(s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}))
    train_accuracies.append(accuracy_score(y_train, s.run(classes, {input_X: X_train_flat})))
    valid_accuracies.append(accuracy_score(y_val, s.run(classes, {input_X: X_val_flat})))

f_1 = visuazlize_loss_accuracy(train_losses, val_losses, train_accuracies, valid_accuracies)
plt.show()


# MLP with hidden layers
# hidden1 = tf.layers.dense(inputs, 256, activation=tf.nn.sigmoid)
tf.reset_default_graph()
s = tf.Session()
input_X = tf.placeholder('float32', shape=(None, 784), name='input_X')
input_y = tf.placeholder('float32', shape=(None, 10), name='input_y')
hidden_1 = tf.layers.dense(input_X, 256, activation=tf.nn.sigmoid)
hidden_2 = tf.layers.dense(hidden_1, 256, activation=tf.nn.sigmoid)
logits = tf.layers.dense(hidden_2, 10)
probas = tf.nn.softmax(logits)
classes = tf.argmax(probas, axis=1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits))
step = tf.train.AdamOptimizer().minimize(loss)

s.run(tf.global_variables_initializer())

train_losses = []
val_losses = []
train_accuracies = []
valid_accuracies = []

BATCH_SIZE = 512
EPOCHS = 40
for epoch in range(EPOCHS):
    batch_losses = []
    for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE):
        _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start+BATCH_SIZE],
                                input_y: y_train_oh[batch_start:batch_start+BATCH_SIZE]})
        batch_losses.append(batch_loss)
    train_losses.append(np.mean(batch_losses))
    val_losses.append(s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}))
    train_accuracies.append(accuracy_score(y_train, s.run(classes, {input_X: X_train_flat})))
    valid_accuracies.append(accuracy_score(y_val, s.run(classes, {input_X: X_val_flat})))

f_2 = visuazlize_loss_accuracy(train_losses, val_losses, train_accuracies, valid_accuracies)
plt.show()

