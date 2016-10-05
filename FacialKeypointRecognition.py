""" facial keypoint recognition with a convolutional network """

import sys
import os
from concurrent.futures import ThreadPoolExecutor

from argparse import ArgumentParser

import numpy as np
import pickle

from sklearn.utils import shuffle

# import lasagne
# import theano
# import theano.Tensor as T
# from lasagne.nonlinearities import leaky_rectify, softmax
from pandas.io.parsers import read_csv

import networks

class FacialKeypointRecognition:
    """
    facial keypoint recognition

    this class builds a convolutional network to find facial keypoints in
    pictures of faces
    """

    ftrain = './data/training.csv'
    ftest = './data/test.csv'
    fIdList = './data/IdList.csv'
    fOutputList = './data/SampleSubmission.csv'
    fOutFile = './data/solution.csv'

    X_train, y_train = [], []
    X_test = []

    prediction = []

    def __init__(self, network=networks.simpleNetwork):
        """
        initialize and run the network
        """

        if type(network) == type:
            self.network = network()
        else:
            self.network = network


    def loadData(self, *args, **kwargs):
        """
        load the images

        the images are given as two csv-files. The training set contains the
        images and up to 30 columns of x- and y-coordinates for the keypoints.
        The test-set only contains the images
        """

        self.X_train, self.y_train = self.load(*args, **kwargs)
        self.X_test, _ = self.load(test=True, *args, **kwargs)


    def load(self,
             test=False,
             cols=None,
             dropMissing=True,
             rescale=True,
             reshape=False,
             rs=42
            ):
        """
        loads one dataset

        *test* - if true, load data from ftest, otherwise load data from ftrain
        *cols* - pass a list of cols if you're only interested in a subset of
            the target columns
        *dropMissing* - by default only columns without missing values are used
        *rescale* - by default pixel values are rescaled to [0, 1]
        *reshape* - by default images are given as one vector instead of a
            two-dimensional array
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
            y = y.astype(np.float32)

            # shuffle training data
            X, y = shuffle(X, y, random_state=rs)
        else:
            y = None

        return X, y


    def fit(self):
        """train the network with the training data"""

        self.network.fit(self.X_train, self.y_train)


    def predict(self):
        """predict the target values for the testset"""

        self.prediction = self.network.predict(self.X_test)


    def savePrediction(self):
        """save the predicted coordinates into a csv file to upload"""

        # transform predictions
        prediction = self.prediction * 48 + 48
        prediction = prediction.clip(0, 96)

        # read id list
        outputset = read_csv(os.path.expanduser(self.fIdList))

        # get needed predictions
        outputPrediction = []
        for i in range(len(outputset)):
            outputPrediction.append(prediction[outputset['ImageId'][i]-1,
                outputset['FeatureName'][i]-1])

        # read output list
        outputset = read_csv(os.path.expanduser(self.fOutputList))

        # fill output list with predictions
        outputset['Location'] = outputPrediction

        # write output list to disk
        outputset.to_csv(self.fOutFile, index=False)


    def saveState(self, filename='network', *, retries=5):
        """save the learned state of the network into a pickle-file"""

        full_filename = "{}.pickle".format(filename)
        hist_filename = "{}_history.pickle".format(filename)

        try:
            with open(full_filename, 'wb') as file:
                pickle.dump(self.network, file, -1)
            with open(hist_filename, 'wb') as file:
                pickle.dump(self.network.network.train_history_, file, -1)
        except RecursionError:
            if retries > 0:
                oldLimit = sys.getrecursionlimit()
                limit = oldLimit * 100
                print("SaveState: Recursion limit of {} reached. Trying again with {}".format(
                    oldLimit, limit))
                sys.setrecursionlimit(limit)
                self.saveState(filename, retries=retries-1)
                sys.setrecursionlimit(oldLimit)
            else:
                print("SaveState: Recursion limit reached. Maximum tries exceeded, giving up")
                print(RecursionError)


    def loadState(self, filename='network.pickle'):
        """load a earlier saved state of the network from a pickle-file"""

        with open(filename, 'rb') as file:
            self.network = pickle.load(file)

    def loadStateAndData(self):
        """
        load both state and data in parallel
        """

        with ThreadPoolExecutor(max_workers=2) as e:
            e.submit(self.loadState, 'net6.pickle')
            e.submit(self.loadData, reshape=True)

def main():
    """
    main function to load the data, train a network on the data and predict
    the testset with the trained network
    """

    # commandline arguments
    # Arguments starting with '-' are optional,
    # nargs='?' = use one following argument as value
    ap = ArgumentParser()
    ap.add_argument('--picklefile', nargs='?')
    ap.add_argument('--epochs', nargs='?', type=int)
    args = ap.parse_args()

    fkr = FacialKeypointRecognition(networks.convolutionalNetwork(args.epochs))

    if args.picklefile:
        # we have a pickle-file that we want to reuse
        fkr.loadState(args.picklefile)

    if args.epochs:
        fkr.network.network.max_epochs = args.epochs

    fkr.loadData(reshape=True)
    fkr.fit()
    fkr.predict()
    fkr.saveState()
    fkr.savePrediction()

# only run when loaded as top file
if __name__ == "__main__":
    main()
