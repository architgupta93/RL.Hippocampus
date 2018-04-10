import numpy as np
import random

def setupPlaceFields(maze, n_place_fields):
    """
    Take a maze and setup place fields for it
    """
    # Get all the locations in the maze
    states   = maze.getStates()
    n_states = len(states)

    # Place fields cover 10% of the maze on an average
    mean_pf_size   = np.sqrt(n_states) * 0.10
    pf_variability = 1.0

    # Select n_place_fields among these to the centers of the place fields
    pf_centers = random.sample(states, n_place_fields)
    pf_sizes   = mean_pf_size + pf_variability * np.random.randn(n_states,)

    # Create and return the place fields
    pfs = []
    for it in range(n_place_fields):
        pfs.append(PlaceField(pf_centers[it], pf_sizes[it]))

    return pfs

def viewPlaceFields(maze, place_fields):
    """
    Visualize the place field activity
    """

    raise NotImplementedError

class PlaceField(object):
    POP_MEAN_FIRING_RATE = 5

    def __init__(self, center, field_size):
        self._center_x = center[0]
        self._center_y = center[1]
        self._field_size = field_size
        self._mean_firing_rate = self.POP_MEAN_FIRING_RATE
        return

    def getActivity(self, pos):
        """
        Given the current position (px, py) of the animal, return the place
        cell activity. The place cell activity is a Gaussian function.
        """

        dist_x   = self._center_x - pos[0]
        dist_y   = self._center_y - pos[1]

        # Assuming spherically symmetric fields for now. This can be changed later on.
        dist_euc = (dist_x * dist_x) + (dist_y * dist_y)
        try:
            activity = self._mean_firing_rate * np.exp(-dist_euc/self._field_size)
            return activity
        except Exception as err:
            print(err)
            return 0
        