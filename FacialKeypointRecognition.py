""" facial keypoint recognition with a convolutional network """

import os
import numpy as np

from sklearn.utils import shuffle

# import lasagne
# import theano
# import theano.Tensor as T
# from lasagne.nonlinearities import leaky_rectify, softmax
from lasagne.updates import nesterov_momentum
from lasagne import layers
from nolearn.lasagne import NeuralNet
from pandas.io.parsers import read_csv

class FacialKeypointRecognition:
    """
    facial keypoint recognition

    this class builds a convolutional network to find facial keypoints in
    pictures of faces
    """

    ftrain = './data/training.csv'
    ftest = './data/test.csv'

    X_train, y_train = [], []
    X_test = []

    def __init__(self):
        """
        initialize and run the network
        """

        self.Network = Network()


    def loadData(self):
        """
        load the images

        the images are given as two csv-files. The training set contains the
        images and up to 30 columns of x- and y-coordinates for the keypoints.
        The test-set only contains the images
        """

        self.X_train, self.y_train = self.load()
        self.X_test = self.load(test=True)


    def load(self,
             test=False,
             cols=None,
             dropMissing=False,
             rescale=True,
             reshape=True,
             rs=42
            ):
        """
        loads one dataset

        *test* - if true, load data from ftest, otherwise load data from ftrain
        *cols* - pass a list of cols if you're only interested in a subset of
            the target columns
        *dropMissing* - by default only columns without missing values are used
        *rescale* - by default pixel values are rescaled to [0, 1]
        *reshape* - by default images are given as a two-dimensional array
            instead of one vector
        *rs* - random state for repeatability
        """

        # load the dataset with pandas
        fname = self.ftest if test else self.ftrain
        df = read_csv(os.path.expanduser(fname))

        # images are pixel values separated by space
        # convert the values to numpy arrays
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        # get a subset of colums if asked for
        if cols:
            df = df[list(cols) + ['Image']]

        # only use cols without missing values
        if dropMissing:
            df = df.dropna()

        # scale pixel values
        if rescale:
            X = np.vstack(df['Image'].values) / 255.
            X = X.astype(np.float32)

        # reshape to two-dimensional array
        if reshape:
            X = X.reshape(-1, 1, 96, 96)

        # ftrain has target values
        if not test:
            y = df[df.columns[:-1]].values

            # scale target coordinates to [-1, 1]
            if rescale:
                y = (y - 48) / 48
            y = y.astype(np.int32)

            # shuffle training data
            X, y = shuffle(X, y, random_state=rs)
        else:
            y = None

        return X, y


class Network:
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
            layers[
                ('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        input_shape=(None, 9216),
        hidden_num_units=100,
        output_nonlinearity=None,
        output_num_units=30,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,
        max_epochs=400,
        verbose=1,
        )


    def fit(self, X, y):
        """use the training set to fit the network"""

        pass


    def predict(self, X):
        """predict the targets after the network is fit"""

        pass


def main():
    """
    main function to load the data, train a network on the data and predict
    the testset on the with the trained network
    """

    fkr = FacialKeypointRecognition()
    fkr.loadData()
    fkr.Network()

    fkr.Network.fit(fkr.X_train, fkr.y_train)


# only run when loaded as top file
if __name__ == "__main__":
    main()
