"""augmentation.py - functions and classes for data augmentation"""

import numpy as np
from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    """
    this class flips the image and the 
    """
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

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

   

def histogrammStreching(data, border=(0,1)):
        """
        strech data from 5% to 95% percent percentile
        """

        data = np.array(data)

        # calculate mean
        mean = np.mean(data, axis = 1)

        # calculate standard deviation
        sigma = np.std(data, axis = 1)

        # correct data
        data = (data-(mean-2*sigma))*(border[1] - border[0])/(4*sigma) + border[0]

        # return reformed list
        return data.tolist()
