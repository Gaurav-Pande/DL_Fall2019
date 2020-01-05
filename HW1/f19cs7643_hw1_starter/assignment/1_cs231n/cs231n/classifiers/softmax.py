import numpy as np
from random import shuffle
# import torch
# from torch.nn import Softmax

def softmax(X):
  """
  Softmax of X, along columns
  """
  X_new = X - X.max(0)
  exp = np.exp(X_new)
  return exp/exp.sum(0)

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  C = W.shape[0]
  D = W.shape[1]
  N = X.shape[1]

  Z = np.dot(W,X) # C*N
  # softmax_nn = Softmax(dim = 0)
  # softmax_out = softmax_nn(torch.from_numpy(Z)).numpy()
  softmax_out = softmax(Z) # C*N - softmax outputs

  # print(softmax_out[y, np.arange(N)])
  # print(np.log(softmax_out[y, np.arange(N)]))
  loss = -(1/N)*np.sum(np.log(softmax_out[y, np.arange(N)] + 1e-6)) # Softmax loss
  # loss *= -1/N
  # print("Loss", loss)
  loss += reg*np.sum(W*W) # Regularization term 

  term_one_dW = softmax_out
  term_one_dW[y, np.arange(N)] -= 1 
  dW = (1/N)*np.dot(term_one_dW, X.T) # C x N, N x D = C x D
  dW += 2*reg*W
  # print(dW)

  return loss, dW

if __name__ == '__main__':
  X = np.array([[1,2,3],[4,5,6]])
  print(X[1])
  # print(softmax(X))
  # W = np.array([[1,2,3],[4,5,6]])
  # y = np.array([0,1,1])
  # W[y, np.arange(3)] -= 1 
  # print(W)
  
  # print(np.sum(W*W))
  C = 3
  # D = 5
  # N = 2
  # X = np.array([[1,2], [32,4], [15,6], [7,28], [9,10]])
  # W = np.ones((C, D))
  # Y = np.array([0, 2])

  # print(X.shape)
  # print(Y.shape)
  # print(W.shape)
  # softmax_loss_vectorized(W, X, Y, 0.2)