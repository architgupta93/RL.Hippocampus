import numpy as np

def setupPlaceFields(maze):
    """
    Take a maze and setup place fields for it
    """


class PlaceField(object):
    POP_MEAN_FIRING_RATE = 5

    def __init__(self, center_x, center_y, field_size):
        self._center_x = center_x
        self._center_y = center_y
        self._field_size = field_size
        self._mean_firing_rate = self.POP_MEAN_FIRING_RATE
        return

    def getActivity(self, px, py):
        """
        Given the current position (px, py) of the animal, return the place
        cell activity. The place cell activity is a Gaussian function.
        """

        dist_x   = self._center_x - px
        dist_y   = self._center_y - py

        # Assuming spherically symmetric fields for now. This can be changed later on.
        dist_euc = (dist_x * dist_x) + (dist_y * dist_y)
        return self._mean_firing_rate * np.exp(dist_euc/self._field_size)