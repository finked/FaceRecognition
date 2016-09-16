""" networks.py - the networks that are used for facial keypoint recognition """

import signal

import numpy as np
import theano

from lasagne.updates import nesterov_momentum
from lasagne import layers
from nolearn.lasagne import NeuralNet

import augmentation

class simpleNetwork:
    """
    this is the neuronal network that is trained and used to predict targets

    this class should be able to be used as a simple shallow network or as a
    deep convolutional network.
    """

    # the neural network
    network = []

    def __init__(self):
        """set up the network with its layers"""

        self.network = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer)],
            input_shape=(None, 9216),
            hidden_num_units=100,
            output_nonlinearity=None,
            output_num_units=30,

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            regression=True,
            max_epochs=400,
            # verbose=1,
        )


    def fit(self, X, y):
        """use the training set to fit the network"""

        return self.network.fit(X,y)


    def predict(self, X):
        """predict the targets after the network is fit"""

        return self.network.predict(X)


class convolutionalNetwork:
    """
    a convolutional network as seen in the example from kaggle
    """

    network = []

    # this variable is read after each epoch
    again = True

    def __init__(self, epochs=None):

        if not epochs:
            epochs = 2000

        self.network = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('dropout3', layers.DropoutLayer),
                ('hidden4', layers.DenseLayer),
                ('dropout4', layers.DropoutLayer),
                ('hidden5', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
            input_shape=(None, 1, 96, 96),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
            dropout1_p=0.1,
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
            dropout2_p=0.2,
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
            dropout3_p=0.3,
            hidden4_num_units=500,
            dropout4_p=0.5,
            hidden5_num_units=500,
            output_num_units=30, output_nonlinearity=None,

            # update_learning_rate=0.01,
            # update_momentum=0.9,

            # data augmentation by flipping the images
            batch_iterator_train=augmentation.FlipBatchIterator(batch_size=128),

            update_learning_rate=theano.shared(float32(0.03)),
            update_momentum=theano.shared(float32(0.9)),

            on_epoch_finished=[
                AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                AdjustVariable('update_momentum', start=0.9, stop=0.999),
                checkAgain(self),
                ],
            regression=True,
            max_epochs=epochs,
            verbose=1,
            )

    def fit(self, X, y):

        # handle the interrupt signal gracefully
        # (by stopping after the current epoch)
        signal.signal(signal.SIGINT, self.handle_break)

        return self.network.fit(X,y)

    def predict(self, X):

        return self.network.predict(X)

    def handle_break(self, signum, frame):
        """
        this function handles the siginterrupt by setting the variable 'again'
        to false
        """

        print("\ninterrupt signal received. Stopping after the current epoch")
        self.again = False


def float32(k):
    return np.cast['float32'](k)


class checkAgain(object):
    """
    helper function to stop the computation if requested by the user
    """

    def __init__(self, network):
        self.network = network

    def __call__(self, nn, train_history):
        if not self.network.again:
            # reset so we can run again!
            self.network.again = True
            raise StopIteration()

class AdjustVariable(object):
    """
    Helper function to adjust a variable inside a network after each epoch
    """

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        if epoch < len(self.ls):
            new_value = float32(self.ls[epoch - 1])
            getattr(nn, self.name).set_value(new_value)
