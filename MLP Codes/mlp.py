import numpy as np

name_func = 'relu'


def act_fun(x, name_func):
    if name_func == 'tanh':
        return np.tanh(x)

    elif name_func == 'sigmoid':
        return 1 / (1 + np.exp(-x))

    elif name_func == 'relu':
        return np.maximum(x, 0)

    elif name_func == 'linear':
        return x


def d_act_fun(x, name_func):
    ''' Derivative of activation function above '''
    if name_func == 'tanh':
        return 1.0 - x ** 2

    elif name_func == 'sigmoid':
        return x * (1 - x)

    elif name_func == 'relu':
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    elif name_func == 'linear':
        return 1


class MLP():
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0] + 1))
        # Hidden layer(s) + output layer
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i + 1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ] * len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * Z - 1) * 0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = act_fun(np.dot(self.layers[i - 1], self.weights[i - 1]), name_func)

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error * d_act_fun(self.layers[-1], name_func)
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * d_act_fun(self.layers[i], name_func)
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error ** 2).sum()
