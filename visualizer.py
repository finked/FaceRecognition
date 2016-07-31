"""
visualizer

methods to get a better understanding of the data we are using
"""

# TODO:
# - showImage / showImages: generalize magic numbers

import numpy as np
import matplotlib.pyplot as plt

def showImage(image, prediction=None, dimensions=None, axis=None, cmap='gray'):
    """
    show one image

    image        - pixel data to draw the image from
    [prediction] - facial keypoints to draw over the image
    [dimensions] - dimensions of the image to reshape. If not set, a square
                   image is assumed
    [axis]       - an subplot object to draw into. If not set, the image is
                   plotted directly
    [cmap]       - color map. Default is gray.
    """

    if not dimensions:
        dim = int(np.sqrt(len(image)))
        dimensions = (dim, dim)

    img = image.reshape(dimensions[0], dimensions[1])

    if axis:
        axis.imshow(img, cmap=cmap)
        if prediction is not None:
            axis.scatter(prediction[0::2] * 48 + 48,
                         prediction[1::2] * 48 + 48,
                         marker='x',
                         s=80
                        )
    else:
        plt.imshow(img, cmap=cmap)
        if prediction is not None:
            plt.scatter(prediction[0::2] * 48 + 48,
                        prediction[1::2] * 48 + 48,
                        marker='x',
                        s=80
                       )
        plt.show()


def showImages(images, predictions=None, imagesInRow=4):
    """
    show multiple images

    this creates a grid with the images.
    The standard number of Images in a row is 4.
    If `imagesInRow` is 0, then all images are printed in one row.
    If predictions are given, they are overayed over the image
    """

    imageCount = len(images)
    if imagesInRow is 0:
        imagesInRow = imageCount
    if imageCount < imagesInRow:
        imagesInRow = imageCount
    columns = np.ceil(imageCount / imagesInRow)

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(imageCount):
        axis = fig.add_subplot(columns, imagesInRow, i+1, xticks=[], yticks=[])
        if predictions is not None:
            showImage(images[i], predictions[i], axis=axis)
        else:
            showImage(images[i], axis=axis)

    plt.show()
