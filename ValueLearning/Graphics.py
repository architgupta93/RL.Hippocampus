import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from MotionAnimation.PY import data_types as GR
import numpy as np

class MazeCanvas(object):
    """
    Used for visualizing the current view of the state space
    """

    def __init__(self, maze):
        self._anim_obj = GR.Trajectory__2D()

        # Fake timestamp needed by the trajectory classes
        self._t_stamp  = 0

        maze_bounds    = maze.getBounds()
        self._min_x    = maze_bounds[0]
        self._max_x    = maze_bounds[2]
        self._min_y    = maze_bounds[1]
        self._max_y    = maze_bounds[3]
        # plt.ion()
    
    def visualizePlaceFields(self, place_fields):
        """
        Show place cell activity for all the positions on the maze
        """

        x_locs = range(self._min_x, self._max_x)
        y_locs = range(self._min_y, self._max_y)
        activity = np.zeros((len(x_locs), len(y_locs)), dtype=float)
        for xi, px in enumerate(x_locs):
            for yj, py in enumerate(y_locs):
                activity[xi, yj] = sum([pf.getActivity((px, py)) for pf in place_fields])

        X, Y = np.meshgrid(x_locs, y_locs)
        # 3D Figure, needs some effort
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, activity, cmap=plt.get_cmap(name='viridis'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('A(x, y)')
        plt.show()

    def update(self, next_state):
        self._anim_obj.update(self._t_stamp, next_state[0], next_state[1])
        self._anim_obj.plotTimedTR(object_type='point')
        self._t_stamp += 1
    
    def plotValueFunction(self, place_fields, critic):
        n_fields = len(place_fields)
        centers  = np.zeros((n_fields, 2), dtype=float)
        for idx, pf in enumerate(place_fields):
            pf_center = pf.getCenter()
            centers[idx, 0] = pf_center[0]
            centers[idx, 1] = pf_center[1]

        # TODO: This needs to be generalized to take non-integer locations?
        x_locs = range(self._min_x, self._max_x)
        y_locs = range(self._min_y, self._max_y)
        values = np.zeros((len(x_locs), len(y_locs)), dtype=float)
        for xi, px in enumerate(x_locs):
            for yj, py in enumerate(y_locs):
                pf_activity = [pf.getActivity((px, py)) for pf in place_fields]
                values[xi, yj] = critic.getValue(pf_activity)
        
        X, Y = np.meshgrid(x_locs, y_locs)

        # 3D Figure, needs some effort
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, values)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('V(x, y)')
        plt.show()

        # TODO: On top of this plot, we should add the place field locations

def plot(*args):
    plt.plot(*args)
    plt.show()