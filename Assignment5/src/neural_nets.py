from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class NeuralNet(object):
  """
  The network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
#     self.params['Wh'] = std * np.random.randn(128, 64)
#     self.params['bh'] = np.zeros(64)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
#     Wh, bh = self.params['Wh'], self.params['bh']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    
#     print(W1.shape)

    # Compute the forward pass
    scores = None

    # FC1 layer.
    fc1_activation = np.dot(X, W1) + b1

    # Relu layer.
    relu_1_activation = fc1_activation
    relu_1_activation[relu_1_activation < 0] = 0
    
    fc2_activation = np.dot(relu_1_activation, W2) + b2

#     relu_2_activation = fc2_activation
#     relu_2_activation[relu_2_activation < 0] = 0

    # FC2 layer.
#     fc3_activation = np.dot(relu_2_activation, W2) + b2

    # Output scores.
    scores = fc2_activation

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0

    # Stability fix for softmax scores
    shift_scores = scores - np.max(scores, axis=1)[..., np.newaxis]

    # Calculate softmax scores.
    softmax_scores = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]
#     softmax_scores = 1 / (1 + np.exp(shift_scores))

    # Calculate our cross entropy Loss.
    correct_class_scores = np.choose(y, shift_scores.T) # Size N vector
    loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
    loss = np.sum(loss)

    # Average the loss & add the regularisation loss: lambda*sum(weights.^2).
    loss /= N
    loss += reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(b1*b1) + np.sum(b2*b2))

    # Backward pass: compute gradients
    grads = {}

    # Calculate dSoft - the gradient wrt softmax scores.
    dSoft = softmax_scores
    dSoft[range(N),y] = dSoft[range(N),y] - 1
    dSoft /= N  # Average over batch.

#     # Backprop dScore to calculate dW2 and add regularisation derivative.
#     dW2 = np.dot(relu_2_activation.T, dSoft)
#     dW2 += 2*reg*W2
#     grads['W2'] = dW2

#     # Backprop dScore to calculate db2.
#     db2 = dSoft * 1
#     grads['b2'] = np.sum(db2, axis=0)

#     # Calculate dx2 and backprop to calculate dRelu2
#     dx2 = np.dot(dSoft, W2.T)
#     relu2_mask = (relu_2_activation > 0)
#     dRelu2 = relu2_mask * dx2
    
#     # Backprop dRelu2 to calculate dWh and add regularization derivative
#     dWh = np.dot(relu_1_activation.T, dRelu2)
#     dWh += 2*reg*Wh
#     grads['Wh'] = dWh

#     # Backprop dRelu2 to calculate dbh
#     dbh = dRelu2 * 1
#     grads['bh'] = np.sum(dbh, axis=0)

#     # Calculate dx2 and backprop to calculate dRelu1.
#     dx1 = np.dot(dRelu2, Wh.T)
#     relu1_mask = (relu_1_activation > 0)
#     dRelu1= relu1_mask*dx1

    # Backprop dScore to calculate dW2 and add regularisation derivative.
    dW2 = np.dot(relu_1_activation.T, dSoft)
    dW2 += 2*reg*W2
    grads['W2'] = dW2

    # Backprop dScore to calculate db2.
    db2 = dSoft * 1
    grads['b2'] = np.sum(db2, axis=0)

    # Calculate dx2 and backprop to calculate dRelu1.
    dx2 = np.dot(dSoft, W2.T)
    relu_mask = (relu_1_activation > 0)
    dRelu1= relu_mask*dx2

    # Backprop dRelu1 to calculate dW1 and add regularisation derivative.
    dW1 = np.dot(X.T, dRelu1)
    dW1 += 2*reg*W1
    grads['W1'] = dW1

    # Backprop dRelu1 to calculate db1.
    db1 = dRelu1 * 1
    grads['b1'] = np.sum(db1, axis=0)

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      # Only take the first 'batch_size' elements.
      random_batch = np.random.permutation(num_train)[0:batch_size]
      X_batch = X[random_batch,...]
      y_batch = y[random_batch]

      # Compute loss and gradients using the current minibatch.
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      # Vanilla gradient descent update.
      self.params['W1'] += -grads['W1']*learning_rate
      self.params['b1'] += -grads['b1']*learning_rate
#       self.params['Wh'] += -grads['Wh']*learning_rate
#       self.params['bh'] += -grads['bh']*learning_rate
      self.params['W2'] += -grads['W2']*learning_rate
      self.params['b2'] += -grads['b2']*learning_rate

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    y_pred = None

    # Get the index of highest score, this is our predicted class.
    y_pred = np.argmax(self.loss(X), axis=1)

    return y_pred