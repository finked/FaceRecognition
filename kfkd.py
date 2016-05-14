"""
kfkd.py

facial keypoint recognition - kaggle learning challenge
"""

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from matplotlib import pyplot

# our datasets
FTRAIN = '~/Projects/FaceRecognition/data/training.csv'
FTEST = '~/Projects/FaceRecognition/data/test.csv'


def load(test=False, cols=None):
    """
    Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the target
    columns.
    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) #load pandas dataframe

    # The image column has pixel values separated by space
    # convert the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # get a subset of columns
    if cols:
        df = df[list(cols) + ['Image']]

    # print the number of values for each column
    print(df.count())

    # drop all rows that are missing values in them
    df = df.dropna()

    # scale pixel values to [0, 1]
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    # only FTRAIN has any target columns
    if not test:
        y = df[df.columns[:-1]].values
        # scale target coordinates to [-1, 1]
        y = (y - 48) / 48
        # shuffle train data
        X ,y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

net1 = NeuralNet(
    # three layers: one hidden layer
    layers=[
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    # 96x96 input pixels per batch
    input_shape=(None, 9216),
    # number of units in hidden layer
    hidden_num_units=100,
    # output layer uses identity function
    output_nonlinearity=None,
    # 30 target values
    output_num_units=30,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    # flag to indicate we're dealing with regression problem
    regression=True,
    # we want to train this many epochs
    max_epochs=400,
    verbose=1,
    )

net1.fit(X, y)

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])

pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()
