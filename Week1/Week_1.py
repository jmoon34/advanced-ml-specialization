import numpy as np
import matplotlib.pyplot as plt
import sys


with open('train.npy', 'rb') as fin:
    X = np.load(fin)
with open('target.npy', 'rb') as fin:
    y = np.load(fin)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
#plt.show()



def expand(X):
    X_expanded = np.zeros((X.shape[0], 6))
    X_expanded[:, 0] = X[:, 0]
    X_expanded[:, 1] = X[:, 1]
    X_expanded[:, 2] = X[:, 0] ** 2
    X_expanded[:, 3] = X[:, 1] ** 2
    X_expanded[:, 4] = X[:, 0] * X[:, 1]
    X_expanded[:, 5] = 1
    return X_expanded

X_expanded = expand(X)

def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above
    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """
    return 1 / (1 + np.exp(-np.dot(X, w)))

dummy_weights = np.linspace(-1, 1, 6)
ans_part1 = probability(X_expanded[:1, :], dummy_weights)[0]
print("ans_part1:", ans_part1)

def compute_loss(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function L using formula above.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    p = probability(X, w)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

ans_part2 = compute_loss(X_expanded, y, dummy_weights)
print("ans_part2:", ans_part2)


def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    return np.dot(X.T, probability(X, w) - y) / X.shape[0]

ans_part3 = np.linalg.norm(compute_grad(X_expanded, y, dummy_weights))
print("ans_part3:", ans_part3)


from IPython import display
h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()

visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])


# Mini-batch gradient

np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])
eta = 0.1
n_iter = 1000
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 100 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    w = w - eta * compute_grad(X_expanded[ind, :], y[ind], w)

visualize(X, y, w, loss)
plt.clf()
ans_part4 = compute_loss(X_expanded, y, w)
print("ans_part4:", ans_part4)

#SGD with momentum
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])
eta = 0.05
alpha = 0.9
nu = np.zeros_like(w)
n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i% 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    nu = alpha * nu + eta * compute_grad(X_expanded[ind, :], y[ind], w)
    w = w - nu

visualize(X, y, w, loss)
plt.clf()


#RMSprop
np.random.seed(42)

w = np.array([0, 0, 0, 0, 0, 1.])

eta = 0.1 # learning rate
alpha = 0.9 # moving average of gradient norm squared
g2 = None # we start with None so that you can update this value correctly on the first iteration
eps = 1e-8

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12,5))
for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # TODO:<your code here>

visualize(X, y, w, loss)
plt.clf()