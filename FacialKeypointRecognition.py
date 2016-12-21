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
from pandas import DataFrame

import networks
from augmentation import histogrammEqualization


class FacialKeypointRecognition:
    """
    facial keypoint recognition

    this class builds a convolutional network to find facial keypoints in
    pictures of faces
    """

    ftrain = './data/training.csv'
    ftest = './data/test.csv'
    fOutFile = './data/solution.csv'

    fOutputList = './data/SampleSubmission_30.csv'
    fIdList = './data/IdList_30.csv'

    fFeatureNames = './data/FeatureNames.csv'

    # for network 8 we only use a subsample of columns
    cols = None
    cols8 = ["left_eye_center_x",
             "left_eye_center_y",
             "right_eye_center_x",
             "right_eye_center_y",
             "nose_tip_x",
             "nose_tip_y",
             "mouth_center_bottom_lip_x",
             "mouth_center_bottom_lip_y"]

    feature_mapping, id_mapping = {}, {}
    X_train, y_train = [], []
    X_test = []

    prediction = []

    def __init__(self, network=networks.simpleNetwork):
        """
        initialize and run the network
        """

        if isinstance(network, networks.network):
            self.network = network
        else:
            self.network = network()

    def setData(self, number):
        """
        set data and solution filenames
        """

        # self.ftrain = './data/training_{}.csv'.format(number)
        # self.ftest = './data/test_{}.csv'.format(number)
        self.fOutFile = './data/solution_{}.csv'.format(number)
        self.fIdList = './data/IdList_{}.csv'.format(number)
        self.fOutputList = './data/SampleSubmission_{}.csv'.format(number)

    def setMapping(self, cols):
        """
        set a mapping from ids/Featurenames to the internal Id of our set

        this mapping is needed when we want to save the set again

        basically we have a list of names with an index that we filter against
        the provided names in 'cols'

        Example:

            id | name
            ---------
             1 | a
             2 | b
             3 | c

             with cols ['a', 'c'] would give the following dict:
             {1: 'a',
              3: 'c'}
        """

        feature_names = read_csv(os.path.expanduser(self.fFeatureNames),
                                 names=['Id', 'Name'],
                                 index_col='Id')
        self.feature_mapping = \
            feature_names[feature_names.Name.isin(cols)].Name.to_dict()

    def setNetwork(self, number):
        """
        set the state for either network 8 or 30
        """
        if number is '8':
            self.cols = self.cols8
            self.network = networks.convolutionalNetwork8()
        else:
            self.cols = None
            self.network = networks.convolutionalNetwork()

    def loadData(self, *args, **kwargs):
        """
        load the images

        the images are given as two csv-files. The training set contains the
        images and up to 30 columns of x- and y-coordinates for the keypoints.
        The test-set only contains the images
        """

        self.X_train, self.y_train = self.load(cols=self.cols, *args, **kwargs)

        # the columns only exist in the trainingsset
        kwargs.pop("cols", None)
        self.X_test, _ = self.load(test=True, *args, **kwargs)

        self.X_train = histogrammEqualization(self.X_train)
        self.X_test = histogrammEqualization(self.X_test)

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

        # if we didn't select any columns, we need to set them here
        # (just take all columns but 'Image')
        if self.cols is None:
            self.cols = df.columns[:-1]

        # only use cols without missing values
        if dropMissing:
            df = df.dropna()

        # create mapping between imageId and position in our numpy array
        # cols['internalId'] = range(len(df))
        # self.id_mapping = df['internalId'].to_dict()
        # # swap index and value
        # self.id_mapping = {v: k for k, v in self.id_mapping.items()}
        self.index = df.index.copy()
        self.index.name = 'imageId'
        print(self.index)

        # scale pixel values
        if rescale:
            X = np.vstack(df['Image'].values) / 255.
            X = X.astype(np.float32)
        else:
            X = np.vstack(df['Image'].values)
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


    def predict(self, X=None):
        """predict the target values for the testset"""

        if X is None:
            X = self.X_test
        self.prediction = self.network.predict(X)


    def savePredictionNew(self, outputfilename=None):
        """
        save the predicted coordinates to a csv file

        this csv file only holds the predicted features
        """

        if outputfilename is None:
            outputfilename = '.data/solution.csv'
        # transform predictions
        prediction = self.prediction * 48 + 48
        prediction = prediction.clip(0, 96)

        # convert from numpy array to pandas dataframe
        # and get our old index and columns back
        outputset = DataFrame(prediction)
        outputset.index = self.index
        outputset.columns = self.cols
        outputset.to_csv(self.fOutFile)

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

    def savePrediction8(self):
        """
        save the predicted coordinates for the 8-feature set into a csv
        file to upload
        """
        # transform predictions
        prediction = self.prediction * 48 + 48
        prediction = prediction.clip(0, 96)

        # read id list
        idset = read_csv(os.path.expanduser(self.fIdList))

        outputPrediction = []
        mapping = {1: 1, 2: 2, 3: 3, 4: 4,
                   5: 5, 6: 6, 7: 7, 8: 8,
                   21: 5, 22: 6, 29: 7, 30: 8}

        for i in range(len(idset)):
            # we only predict the second part of the set of images.
            # so we need to shift by 592
            # TODO(tobias): shift the images in IdList_8.csv
            ImageID = idset['ImageId'][i]-592
            Feature = idset['FeatureName'][i]
            newFeatureId = mapping[Feature]
            outputPrediction.append(prediction[ImageID, newFeatureId-1])

        # read output list
        outputset = read_csv(os.path.expanduser(self.fOutputList))

        # fill output list with predictions
        outputset['Location'] = outputPrediction

        # write output list to disk
        outputset.to_csv(os.path.expanduser(self.fOutFile), index=False)

    def saveState(self, filename='network', foldername='pickle', *, retries=5):
        """save the learned state of the network into a pickle-file"""

        full_filename = "{}/{}.pickle".format(foldername, filename)
        hist_filename = "{}/{}_history.pickle".format(foldername, filename)

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


    def loadState(self, filename='network.pickle', foldername='pickle'):
        """load a earlier saved state of the network from a pickle-file"""

        full_filename = '{}/{}'.format(foldername, filename)
        with open(full_filename, 'rb') as file:
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
    ap.add_argument('--picklefile', nargs='?',
                    help='saved state to resume from')
    ap.add_argument('--network', nargs='?',
                    help='which network to use. '
                    'The only possible values right now are 8 or 30')
    ap.add_argument('--epochs', nargs='?', type=int,
                    help='how many epochs the network should run fitting')
    ap.add_argument('--dataset', nargs='?',
                    help='which dataset to load and use. '
                    'The only possible values right now are 8 or 30')
    ap.add_argument('--outputfilename', nargs='?',
                    help='filename of the prediction output')
    args = ap.parse_args()

    fkr = FacialKeypointRecognition()
    if args.network:
        fkr.setNetwork(args.network)

    if args.picklefile:
        # we have a pickle-file that we want to reuse
        fkr.loadState(args.picklefile)

    # if we loaded a picklefile, it has epochs defined that may not be the same
    # as our argument
    if args.epochs:
        fkr.network.network.max_epochs = args.epochs

    if args.dataset:
        fkr.setData(args.dataset)
    fkr.loadData(reshape=True)
    fkr.fit()
    fkr.predict()
    fkr.saveState()

    fkr.savePredictionNew(args.outputfilename)

# only run when loaded as top file
if __name__ == "__main__":
    main()
