import numpy as np
from cs231n.classifiers.softmax import *

class LinearClassifier:

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: D x N array of training data. Each training point is a D-dimensional
         column.
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    dim, num_train = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = np.random.randn(num_classes, dim) * 0.001

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      # print("Loop")

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      indices = np.random.choice(num_train, batch_size)
      y_batch = y[indices]
      X_batch = X[:, indices]
      # print(X_batch)
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      # print(loss)
      loss_history.append(loss)

      self.W = self.W - learning_rate * grad
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    k = self.W.shape[0]
    output = np.dot(self.W, X)
    output = softmax(output)
    y_pred = [np.random.choice(k, p=softmax_out) for softmax_out in output.T]
    assert len(y_pred) == X.shape[1]
    return y_pred

  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - X_batch: D x N array of data; each column is a data point.
    - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    # print("Came here")
    # # print("X batch", X_batch)
    # loss, gradient = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
    pass


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    # print("Came here")
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

if __name__ == '__main__':
  classifier = Softmax()
  loss_hist = classifier.train(
    X_train,
    y_train,
    learning_rate=1e-3,
    reg=1e-5,
    num_iters=10,
    batch_size=200,
    verbose=False,
  )
