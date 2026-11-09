"""Unit 14: a 2-4-1 neural network learns XOR by backpropagation.

XOR cannot be separated by one straight line, so a single neuron
(logistic regression) fails; one hidden layer is enough.
"""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
rng = np.random.default_rng(465)
W1, b1 = rng.normal(0, 1, (2, 4)), np.zeros((1, 4))
W2, b2 = rng.normal(0, 1, (4, 1)), np.zeros((1, 1))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


lr = 1.0
for epoch in range(1, 5001):
    h = sigmoid(X @ W1 + b1)                  # forward pass: hidden layer
    out = sigmoid(h @ W2 + b2)                # forward pass: output
    loss = -np.mean(y * np.log(out) + (1 - y) * np.log(1 - out))
    d_out = (out - y) / len(X)                # backward pass (chain rule)
    d_W2, d_b2 = h.T @ d_out, d_out.sum(axis=0, keepdims=True)
    d_h = (d_out @ W2.T) * h * (1 - h)
    d_W1, d_b1 = X.T @ d_h, d_h.sum(axis=0, keepdims=True)
    W1, b1 = W1 - lr * d_W1, b1 - lr * d_b1   # gradient descent step
    W2, b2 = W2 - lr * d_W2, b2 - lr * d_b2
    if epoch in (1, 100, 500, 1000, 5000):
        print(f"epoch {epoch:4d}: loss {loss:.4f}")
print("outputs for 00, 01, 10, 11:", out.ravel().round(3))
