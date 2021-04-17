import numpy as np
import random

N_CELLS_PER_FIELD = 10
FIELD_CENTER_JITTER = 0.2

def assignPlaceCells(n_cells, place_fields):
    """
    Take a set of place fields and assign cells to them
    """

    # Right now the setup is exact. Every N_CELLS_PER_FIELD share the same place field
    n_fields = len(place_fields)
    cells_per_field = round(n_cells/n_fields)
    cell_shuffle_assignment = random.sample(range(n_cells), n_cells)
    total_cells_assigned = 0
    cells = np.empty(n_cells, dtype=PlaceCell)

    for cell_cohort in range(n_fields):
        n_cells_in_cohort = 0
        field_size = place_fields[cell_cohort].getFieldSize()
        field_center = place_fields[cell_cohort].getCenter()

        while (total_cells_assigned < n_cells) and (n_cells_in_cohort <  cells_per_field):
            cells[cell_shuffle_assignment[total_cells_assigned]] = PlaceCell(field_center, field_size)
            n_cells_in_cohort += 1
            total_cells_assigned += 1

    return cells

def setupPlaceFields(maze, n_place_fields):
    """
    Take a maze and setup place fields for it
    """
    # Get all the locations in the maze
    states   = maze.getStates()
    n_states = len(states)

    # Select n_place_fields among these to the centers of the place fields
    pf_centers = random.sample(states, n_place_fields)

    # Add some noise to the place field centers
    for field_idx in range(n_place_fields):
        pf_centers[field_idx] = (pf_centers[field_idx][0] + FIELD_CENTER_JITTER * np.random.randn(),
            pf_centers[field_idx][1] + FIELD_CENTER_JITTER * np.random.randn())

    # print("Place field centers...")
    # print(pf_centers)

    # Place fields cover 5% of the maze on an average
    pf_sizes   = np.sqrt(n_states) * 0.05 * np.ones(n_states, dtype=float)

    # Create and return the place fields
    pfs = []
    for it in range(n_place_fields):
        pfs.append(PlaceField(pf_centers[it], pf_sizes[it]))

    return pfs

class PlaceField(object):
    def __init__(self, center=None, field_size=None):
        if center is None:
            center = (np.Inf, np.Inf)

        self._center_x = center[0]
        self._center_y = center[1]
        self._field_size = field_size
        return

    def getCenter(self):
        # TODO: Might have to return the field sizes too!
        return (self._center_x, self._center_y)

    def getFieldSize(self):
        return self._field_size

class PlaceCell(PlaceField):
    POP_MEAN_FIRING_RATE = 0.5
    def __init__(self, center=None, field_size=None):
        """
        We can have multiple cells that represent that same place field (at
        least similar place field). Such an over-representation might help
        learn multiple environments simultaneously.
        """
        super(PlaceCell, self).__init__(center, field_size)
        self._is_non_place = False
        if field_size is None:
            self._is_non_place = True

        self._mean_firing_rate = self.POP_MEAN_FIRING_RATE

    def getActivity(self, pos):
        """
        Given the current position (px, py) of the animal, return the place
        cell activity. The place cell activity is a Gaussian function.
        """
        if self._is_non_place:
            # TODO: We can try out random spiking at the mean firing rate here
            # if the cell is not associated with any particular place.
            return 0

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
        
