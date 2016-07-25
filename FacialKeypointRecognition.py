""" facial keypoint recognition with a convolutional network """

import os
import numpy as np

from sklearn.utils import shuffle

import lasagne
import theano
import theano.Tensor as T
from lasagne.nonlinearities import leaky_rectify, softmax
from pandas.io.parsers import read_csv

class FacialKeypointRecognition:
    """
    facial keypoint recognition

    this class builds a convolutional network to find facial keypoints in
    pictures of faces
    """

    ftrain = './data/training.csv'
    ftest = './data/test.csv'

    def __init__(self):
        """
        initialize and run the network
        """

        pass


    def loadData(self):
        """
        load the images

        the images are given as two csv-files. The training set contains the
        images and up to 30 columns of x- and y-coordinates for the keypoints.
        The test-set only contains the images
        """

        X_train, y_train = self.load()
        X_test = self.load(test=True)


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


def main():
    """
    main function to load the data, train a network on the data and predict
    the testset on the with the trained network
    """

    fkr = FacialKeypointRecognition()
    fkr.loadData()


# only run when loaded as top file
if __name__ == "__main__":
    main()
