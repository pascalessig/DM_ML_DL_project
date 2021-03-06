# -*- coding: utf-8 -*-
"""simple_ann.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pmCOObmO3UIApyK9dXOCRIX2uAEyN1Oq
"""

# imports
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

"""Variable Definition:
layer_dims  -- array containing the amount of neurons in each layer if the nn
parameters -- dictionary holding the weight matrices and bias arrays for each layer
"""

def initialize_weights(layer_dims):
  """
  Inizialization of parameters W_l and b_l which are the weight and bias matrices
  of the corresponding layers l. The weights are initialized according to the
  so called He initialization (He et al.) CITE PAPER which is industry standard.
  This initialization helps the network to converge faster and minimized the risk
  of vanishing and exploiding weights.

  Argument:
  layer_dims -- array of length L which stores the amount of neural units
                per layer

  Return:
  parameters -- dictionary which stores the W_l and b_l values for each layer l
                in the NN

  """

  np.random.seed(123) # include seed for consistent results

  L = len(layer_dims)
  parameters = dict()

  for l in range(1, L):
    parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

  return parameters

def relu(x):
  """
  Apply the rectified linear unit function on the argument x.

  Argument:
  x -- x can be a float or numpy array/matrix

  Return:
  x -- if the value of the element is bigger than 0
  0 -- if the value of the element is zero or negative
  """

  return np.maximum(0,x)

def relu_deriv(x):
  """
  Apply the derivate of the recfitied linear unit on the argument x.

  Argument:
  x -- x can be a float or numpy array/matrix

  Return:
  1 -- if the value of the element is bigger than zero
  0 -- if the value of the element is zero or negative
  """

  return (x > 0) * 1

def sigmoid(x):
  """
  Apply the sigmoid function on the argument x.

  Argument:
  x -- x can be a float or numpy array/matrix

  Return:
  sigmoid(x) -- x transformed by sigmoid funciton
  """

  s = 1/(1+np.exp(-x))

  return s

def sigmoid_deriv(x):
  """
  Apply the derivate of the sigmoid function on the argument x.

  Argument:
  x -- x can be a float or numpy array/matrix

  Return:
  the derivative of the sigmoid funciton applied on x
  """

  s = 1/(1+np.exp(-x))

  ds = s * (1-s)

  return ds

"""Linear calculation for a layer l is calculated by

$$Z_l = W_lA_{l-1} + b_l$$

the non linear part of if calculated by

$$A_l = \sigma(Z_l)$$


where the activation function $\sigma$ is either a Rectified Linear Unit (ReLU),

$$\sigma(x) = \left\{
  \begin{array}{rcl}
  x & \mbox{if} & x > 0 \\
  0 & \mbox{if} & x \leqslant 0
  \end{array}
  \right. $$

which is used within the network, or a sigmoid function,

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

which is only used in the prediction layer.
"""

def forward_propagation(X, parameters, layer_dims, test=False):
  """
  Performs the forward propagation of the data throught the network as
  described above.

  Arguments:
  X          -- the features of batch_size observations (shape(layer_dims[0], batch_size))
  parameters -- the weight matrices W_l (shape(layer_dims[l], layer_dims[l-1]))
                the bias vectors b_l (shape(layer_dims[l], 1))
  layer_dims -- array of length L which stores the amount of neural units
                per layer

  Returns:
  states     -- the Z_l value (shape(layer_dims[l], batch_size))
                the A_l value (shape(layer_dims[l], batch_size))
                variables calculated as desctibed above.
                States are optional and will be created if not given
  prediction -- the A_L matrix which are the predictions of the network
  """
  states = {'A0' : X}
  L = len(layer_dims) - 1 # define L one less to predict sigmoid activation extra

  for l in range(1, L): # internal states for layers 1 - (L-1) because relu activation
    states['Z' + str(l)] = np.dot(parameters["W" + str(l)], states["A" + str(l-1)]) + parameters["b" + str(l)] # linear part of forward propagation
    states['A' + str(l)] = relu(states['Z' + str(l)]) # non-linear activation funciton

  # internal states for layer L because sigmoid function is used for prediction
  states['Z' + str(L)] = np.dot(parameters["W" + str(L)], states["A" + str(L-1)]) + parameters["b" + str(L)]
  states['A' + str(L)] = sigmoid(states['Z' + str(L)])
#  if test:
#      # adjust A_L to not encounter numeric issues in cost function
#      new_1 = 0.00001
#      new_0 = 0.99999
#      states['A' + str(L)][states['A' + str(L)] == 1.] = 0.999999
#      states['A' + str(L)][states['A' + str(L)] == 0.] = 0.000001

  return states, states['A' + str(L)]

"""Because we create the network for classification purposes, we used the cross-entropy loss function in our network to calculate "how right the network predicts". The formula of the cross-entropy loss is:

$$L(\hat{Y}, Y) = -\frac{1}{m} \sum_{i = 1}^m y_i * log(\hat{y}_i) + (1-y_i) * log(1 - \hat{y}_i)$$

where $m$ is the batch size, $\hat{y}_i$ is the predicted label for one observation in the batch and $y_i$ is the true label of the observation in the batch.
"""

def cross_entropy_cost(Y_hat, Y, weights):
  """
  Calculates the average cost per batch using the cross-entropy loss
  as described above.

  Arguments:
  Y     -- array of the true labels for the observations in the batch
  Y_hat -- array of the predicted labels for the observations in the batch

  Returns:
  cost  -- the calculated loss for the batch
  """

  m = Y.shape[1]
  w_min = weights['w_min']
  w_maj = weights['w_maj']

  cost = - (np.dot(Y, np.log(Y_hat.T * w_min)) + np.dot((1-Y), np.log((1-Y_hat).T * w_maj))) / m
  cost = np.squeeze(cost)
  return cost

def backward_propagation(Y, states, parameters, layer_dims, gradients, weights):
  """
  Performes the back propagation of the errors through the network. Calculates
  the gradients of the parameters using the commonly known derivatives for
  weight matrices and bias vectors.

  Arguments:
  Y          -- array of the true labels for the observations in the batch
  states     -- the Z_l values (shape(layer_dims[l], batch_size))
                the A_l values (shape(layer_dims[l], batch_size))
  parameters -- the weight matrices W_l (shape(layer_dims[l], layer_dims[l-1]))
                the bias vectors b_l (shape(layer_dims[l], 1))
  layer_dims -- array of length L which stores the amount of neural units
                per layer

  Returns:
  gradients  -- a dictionary containing the gradients of the corresponding
                parameters

  """

  L = len(layer_dims) - 1
  m = Y.shape[1]
  w_min = weights['w_min']
  w_maj = weights['w_maj']

  gradients['dA' + str(L)] = - (np.divide(Y * w_min, states['A' + str(L)]) - np.divide((1 - Y) * w_maj, 1 - states['A' + str(L)]))
  gradients['dZ' + str(L)] = np.multiply( gradients['dA' + str(L)], sigmoid_deriv(states['Z' + str(L)]) )
  gradients['dW' + str(L)] = np.dot(gradients['dZ' + str(L)], states['A' + str(L-1)].T) / m
  gradients['db' + str(L)] = np.sum(gradients['dZ' + str(L)], axis=1, keepdims=True) / m

  for l in reversed(range(1, L)): # L-1 ... 1
    gradients['dA' + str(l)] = np.dot(parameters['W' + str(l+1)].T, gradients['dZ' + str(l+1)])
    gradients['dZ'+ str(l)] = np.multiply(gradients['dA' + str(l)] , relu_deriv(states['Z' + str(l)]))
    gradients['dW' + str(l)] = np.dot(gradients['dZ' + str(l)], states['A' + str(l-1)].T) / m
    gradients['db' + str(l)] = np.sum(gradients['dZ' + str(l)], axis=1, keepdims=True) / m

  return gradients

"""Update parameters according to the commonly update function:

$$W_l = W_l - \alpha * \frac{\delta}{\delta w_l}L(w_l,b_l)$$
$$b_l = b_l - \alpha * \frac{\delta}{\delta b_l}L(w_l,b_l)$$
"""

def update_parameters(parameters, gradients, layer_dims, learning_rate):
  """
  Updates the parameters of the network by the update rule as defined above.

  Arguments:
  parameters    -- the weight matrices W_l (shape(layer_dims[l], layer_dims[l-1]))
                   the bias vectors b_l (shape(layer_dims[l], 1))
  gradients     -- a dictionary containing the gradients of the corresponding
                   parameters
  layer_dims    -- array of length L which stores the amount of neural units
                   per layer
  learning_rate -- defines the step size of an update step (float)

  Returns:
  parameters    -- the weight matrices W_l (shape(layer_dims[l], layer_dims[l-1]))
                   the bias vectors b_l (shape(layer_dims[l], 1))
  """

  L = len(layer_dims)

  for l in range(1, L):
    parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * gradients['dW' + str(l)]
    parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * gradients['db' + str(l)]

  return parameters

def plot_costs(costs):
  """
  Plots the training costs of the NN on the train and test set.

  Argument:
  costs -- dictionary containing the num_iteration, training cost and test cost
  """
  labels, train_cost, test_cost = zip(*[(l, d['train'], d['test']) for l, d in costs.items()])
  plt.plot(labels, train_cost, 'b', label = 'train cost')
  plt.plot(labels, test_cost, 'r', label = 'test cost')
  plt.legend(bbox_to_anchor=(0.7, 0.95), loc='upper left', borderaxespad=0.)
  plt.show()

def train_network(X, Y, X_test, Y_test, layer_dims, learning_rate = 0.01, num_iterations = 200, c_plot = True, num_checkpoints = 10, learn_adjust = [], params = '', weights = {'w_min':1, 'w_maj':1}):
  """
  Training the network using the functions defined above.

  Arguments:
  X               -- descriptive variables of training set (numpy.array dims [#features, #observations])
  Y               -- described variables of training set (numpy.array dims [1, #observations])
  X_test          -- descriptive variables of test set (numpy.array dims [#features, #observations])
  y_test          -- described variables of test set (numpy.array dims [1, #observations])
  layer_dims      -- array specifying the amount of perceptrons of each layer 1 to L (list)
  learning_rate   -- (optional) the learning rate to use with the network (float)
                     default: learning_rate = 0.01
  num_iterations  -- the amount of iterations used for training (int)
  c_plot          -- (optional) whether the train and test costs should be plot or not (boolean)
                     default: plot_costs = True
  num_checkpoints -- (optional) how often to print training cost and training accuracy (int)
                     default: num_checkpoints = 10


  Returns:
  predictions -- the predictions of the NN on the test data after training
  parameters  -- the parameters of the trained NN
  """

  # initialize metaparameters
  layer_dims.insert(0,X.shape[0])
  costs = dict()
  gradients = dict()
  if params != '':
      parameters = params
  else:
      parameters = initialize_weights(layer_dims)
  denominator = num_iterations / num_checkpoints
  param_dict = dict()

  # train the network for num_iterations iterations
  for i in range(1, num_iterations+1):
    if i in learn_adjust:
        learning_rate = learning_rate * 0.1

    # perform forward_propagation
    states, y_pred = forward_propagation(X, parameters, layer_dims)

    # calculate the cross-entropy loss
    cost = cross_entropy_cost(y_pred, Y, weights)

    # count right predictions
    cnt_correct = np.sum(np.abs((y_pred > 0.5).astype(float) == Y))

    # calculate gradients using backward propagation
    gradients = backward_propagation(Y, states, parameters, layer_dims, gradients, weights)

    # update parameters
    parameters = update_parameters(parameters, gradients, layer_dims, learning_rate)

    # printing checkpoints num_checkpoint times
    if i % denominator == 0:
      # calculate and store measures
      param_dict.update({i:parameters.copy()})
      costs[i] = {'train' : cost}
      acc = cnt_correct / X.shape[1]
      _, test_pred = forward_propagation(X_test, parameters, layer_dims, test=True)
      test_cost = cross_entropy_cost(test_pred, Y_test, weights)
      costs[i].update({'test' : test_cost})

      # print measures
      print('Iteration: %i' %i)
      print('\tTraining Accuracy: %f' %acc)
      print('\tTraining Cost: %f' %cost)
      print('\tTest Cost: %f' %test_cost)


  # plot the costs after training
  if c_plot == True:
    plot_costs(costs)

  # calculate predictions on test set after training
  _, y_score = forward_propagation(X_test, parameters, layer_dims, test=True)
  y_score = (test_pred > 0.5).astype(float)

  return y_score, param_dict, costs

def get_train_test_iris():
  """
  Gets the data of the commonly known iris classification dataset and trims it
  to be able to perform binary classification.

  Returns:
  X_train -- descriptive variables of training set (numpy.array dims [#features, #observations])
  y_train -- described variables of training set (numpy.array dims [1, #observations])
  X_test  -- descriptive variables of test set (numpy.array dims [#features, #observations])
  y_test  -- described variables of test set (numpy.array dims [1, #observations])
  """
  iris = load_iris()
  X = iris.data[:100,:]
  Y = iris.target[:100].reshape(-1,1)
  data = np.concatenate((X,Y), axis = 1)
  np.random.shuffle(data)
  data = data.T
  X_train= data[:4, :80]
  X_test= data[:4, 80:]
  y_train = data[4, :80].reshape(1, -1)
  y_test = data[4, 80:].reshape(1, -1)

  return X_train, X_test, y_train, y_test

def run_iris_test_case(num_iterations = 300, learning_rate = 0.01):
  """
  Runs a test case using the modified iris dataset to check the functionality
  of the network

  Returns:
  test_pred -- the predictions of the NN on the test set after training
  """
  layer_dims = [5,3,1]
  X_train, X_test, y_train, y_test = get_train_test_iris()
  test_pred, _ =  train_network(X_train, y_train, X_test, y_test, layer_dims, learning_rate, num_iterations)
  return test_pred
