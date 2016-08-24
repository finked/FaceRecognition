""" networks.py - the networks that are used for facial keypoint recognition """

import numpy as np

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

    def __init__(self):

        self.network = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('hidden4', layers.DenseLayer),
                ('hidden5', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
            input_shape=(None, 1, 96, 96),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
            hidden4_num_units=500, hidden5_num_units=500,
            output_num_units=30, output_nonlinearity=None,

            update_learning_rate=0.01,
            update_momentum=0.9,

            # data augmentation by flipping the images
            batch_iterator_train=augmentation.FlipBatchIterator(batch_size=128),

            regression=True,
            max_epochs=50,
            verbose=1,
            )

    def fit(self, X, y):

        return self.network.fit(X,y)

    def predict(self, X):

        return self.network.predict(X)
