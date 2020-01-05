import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  N = x.shape[0]
  X = np.reshape(x, (N, -1))
  out = np.dot(X, w) + b
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  dx, dw, db = None, None, None
  x, w, b = cache
  N = x.shape[0]
  X = np.reshape(x, (N, -1))
  dw = np.dot(X.T, dout)
  dx = np.dot(dout, w.T)
  dx = np.reshape(dx, x.shape)
  db = np.sum(dout, axis = 0)
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  mask = x < 0
  out = x.copy()
  out[mask] = 0
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  mask = x < 0
  dx = dout
  dx[mask] = 0
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def convolve(data_point, filter, stride):
  """
  Input:
  - data_point: Shape is (C, H, W) or (H, W)
  - filter: Shape is (C, HH, WW) or (HH, WW)
  - stride: Integer
  Output:
  - Integer whose value is convolution
  """
  output = 0
  r = 0
  if len(filter.shape) == 2:
    # assert data_point.shape == 2
    filter_here = np.reshape(filter, (1, *filter.shape))
    data_point_here = np.reshape(data_point, (1, *data_point.shape))
  elif len(filter.shape) == 3:
    # assert data_point.shape == 3
    filter_here = filter
    data_point_here = data_point
  
  assert filter_here.shape[0] == data_point_here.shape[0]

  C, HH, WW = filter_here.shape
  _, H, W = data_point_here.shape
  output = np.zeros((1 + (H - HH) // stride, 1 + (W - WW) // stride))
  r_index = 0
  while r + HH <= H:
    c = 0
    c_index = 0
    while c + WW <= W:
      value_here = np.sum(filter_here[:, :, :] * data_point_here[:, r:r + HH, c:c + WW])
      output[r_index][c_index] = value_here
      c += stride
      c_index += 1
    r += stride
    r_index += 1
  return output

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  assert x.shape[1] == w.shape[1]
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  out = np.zeros((N, F, 1 + (H + 2 * pad - HH) // stride, 1 + (W + 2 * pad - WW) // stride))
  index_point = 0
  for data_point in x:
    data_point_padded = np.pad(data_point, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values=(0))
    index_filter = 0
    for filter in w:
      value_for_filter = convolve(data_point_padded, filter, stride)
      value_for_filter += b[index_filter]
      out[index_point][index_filter] = value_for_filter
      index_filter += 1
    index_point += 1
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape

  stride = conv_param['stride']
  pad = conv_param['pad']
  
  dw = np.zeros_like(w)

  db = dout.sum(0).sum(1).sum(1)
  
  x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=(0))
  dx_pad = np.zeros_like(x_pad)
  
  for index_point in range(N):
    dout_point = dout[index_point]
    data_point = x[index_point]
    for index_filter in range(F):
      current_filter = w[index_filter]
      dout_filter = dout_point[index_filter]
      height_start = 0
      while height_start  + HH <= H + 2*pad:
        width_start = 0
        while width_start + WW <= W + 2*pad:
          dw[index_filter] +=  x_pad[index_point, :, height_start : height_start + HH, width_start : width_start + WW] * dout_filter[height_start // stride, width_start // stride]
          dx_pad[index_point, :, height_start : height_start + HH, width_start : width_start + WW] += dout_filter[height_start // stride, width_start // stride] * current_filter
          width_start += stride
        height_start += stride
  dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  out = np.zeros((N, C, 1 + (H - pool_height) // stride, 1 + (W - pool_width) // stride))

  for index_point in range(N):
    for index_channel in range(C):
      height_start = 0
      while height_start + pool_height <= H:
        width_start = 0
        while width_start + pool_width <= W:
          out[index_point][index_channel][height_start//stride][width_start//stride] = np.max(x[index_point][index_channel][height_start:height_start + pool_height, width_start:width_start + pool_width])
          width_start += stride
        height_start += stride

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  dx = np.zeros_like(x)
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  for index_point in range(N):
    for index_channel in range(C):
      height_start = 0
      while height_start + pool_height <= H:
        width_start = 0
        while width_start + pool_width <= W:
          max_index = np.argmax(x[index_point][index_channel][height_start:height_start + pool_height, width_start:width_start + pool_width])
          a,b = np.unravel_index(max_index, (pool_height, pool_width))
          dx[index_point][index_channel][a + height_start][b + width_start] = dout[index_point][index_channel][height_start//stride][width_start//stride]
          width_start += stride
        height_start += stride
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx