"""augmentation.py - functions and classes for data augmentation"""

import numpy as np
from nolearn.lasagne import BatchIterator


class FlipBatchIterator(BatchIterator):
    """
    this class flips the image and corresponding target values
    """
    flip_indices = []

    def setFlipList(self, flip_indices):
        self.flip_indices = flip_indices

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, int(bs / 2), replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb


def histogrammStreching(data, border=(0.0, 1.0)):
    """
    strech data from 5% to 95% percent percentile
    """

    print(data.shape)

    data = np.array(data)

    # calculate mean
    mean = np.mean(data, axis=1)
    # mean = 0.5

    # calculate standard deviation
    sigma = np.std(data, axis=1)
    # sigma = 0.3

    print('mean: {}'.format(mean))
    # print('mean shape: {}'.format(len(mean)))
    print('sigma: {}'.format(sigma))
    print('sigma shape: {}'.format(len(sigma)))

    # correct data
    data = ((data.T-(mean-0.5*sigma))*(border[1] - border[0])
            / (1*sigma) + border[0]).T

    # transform float to int type
    # data = data.astype(np.float32)

    # return reformed list
    return data


def histogrammEqualization(data, nbr_bins=256):

    newData = []
    for image in data:
        # get image histogram
        imhist, bins = np.histogram(image.flatten(), nbr_bins, density=True)
        cdf = imhist.cumsum()   # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]   # normalize

        # use linear interpolation of cdf to find new pixel values
        im2 = np.interp(image.flatten(), bins[:-1], cdf)
        im2 = im2/255.
        im2 = im2.astype(np.float32)

        newData.append(im2.reshape(image.shape))

    return np.array(newData)


def ZCSwhitening(data, epsilon=0.1):
    """
    Whiten the data and remove second order structure
    """
    newData = []
    for image in data:
        sigma = np.dot(image, image.T)/image.shape[1]   # Correlation matrix
        U, S, V = np.linalg.svd(sigma)                  # Singular value decomposition
        ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)
        newData.append(np.dot(ZCAMatrix, image))        # Data whitening        
    return newData
