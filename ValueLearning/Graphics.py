import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from MotionAnimation.PY import data_types as GR
import numpy as np

class MazeCanvas(object):
    """
    Used for visualizing the current view of the state space, as well as the
    trajectory taken while traversing it.
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

    def visualizePlaceFields(self, place_cells):
        """
        Show place cell activity for all the positions on the maze
        """

        x_locs = np.linspace(self._min_x, self._max_x-1, 100)
        y_locs = np.linspace(self._min_y, self._max_y-1, 100)
        activity = np.zeros((len(x_locs), len(y_locs)), dtype=float)
        for xi, px in enumerate(x_locs):
            for yj, py in enumerate(y_locs):
                activity[xi, yj] = sum([pf.getActivity((px, py)) for pf in place_cells])

        Y, X = np.meshgrid(y_locs, x_locs)
        # 3D Figure, needs some effort
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, activity, cmap=plt.get_cmap(name='viridis'))
        ax_cmap = cm.ScalarMappable(cmap='viridis')
        ax_cmap.set_array(activity)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('A(x, y)')
        plt.colorbar(ax_cmap)
        plt.show()

    def update(self, next_state):
        # Append the data points to the appropriate trajectory object
        self._anim_obj.update(self._t_stamp, next_state[0], next_state[1])
        self._t_stamp += 1

    def plotTrajectory(self):
        # The -1 is just because of the way range() works. Start is included
        # but end is not. This cuts out the first row and first column of data
        self._anim_obj.setLims((self._min_x-1, self._max_x), (self._min_y-1, self._max_y))
        self._anim_obj.plotStaticTR(object_type='line')
    
    def animateTrajectory(self):
        self._anim_obj.plotTimedTR(object_type='point')
    
    def plotValueFunction(self, place_cells, critic):
        # Scale the number of data points in each dimension independently
        nx_pts = round(20 * (self._max_x - self._min_x))
        ny_pts = round(20 * (self._max_y - self._min_y))
        x_locs = np.linspace(self._min_x, self._max_x, num=nx_pts)
        y_locs = np.linspace(self._min_y, self._max_y, num=ny_pts)

        # Have never quite understood why Meshgrid works in such a wierd way
        values = np.zeros((len(x_locs), len(y_locs)), dtype=float)
        for xi, px in enumerate(x_locs):
            for yj, py in enumerate(y_locs):
                pf_activity = [pf.getActivity((px, py)) for pf in place_cells]
                values[xi, yj] = critic.getValue(pf_activity)
        
        # print(values)
        Y, X = np.meshgrid(y_locs, x_locs)

        # 3D Figure, needs some effort
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, values)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.set_zlabel('V(x, y)')

        # Set the format for the axes ticks
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.show()

        # TODO: On top of this plot, we should add the place field locations

class WallMazeCanvas(MazeCanvas):
    """
    Class extending the functionality of the basic canvas class. This can
    plot additional features like obstacles in the Maze.
    """

    def __init__(self, maze):
        super(WallMazeCanvas, self).__init__(maze)
        self._maze = maze
        return

    def plotTrajectory(self):
        """
        Plots the Maze's structure and overlays the agent's trajectory on it.
        """
        fig = self._anim_obj.getContainer()

        for wall in self._maze.getWalls():
            [xs, ys] = wall.getPlottingData()
            fig.plot(xs, ys)

        self._anim_obj.setLims((self._min_x-1, self._max_x), (self._min_y-1, self._max_y))
        self._anim_obj.plotStaticTR(object_type='line', figure_handle=fig)

def plot(*args):
    # Create a new figure
    new_figure = plt.figure()
    axes = new_figure.add_subplot(111)
    axes.plot(*args)
    # axes.set_xlabel('Trial')
    # axes.set_ylabel('Latency')
    plt.show()

def histogram(data):
    new_figure = plt.figure()
    axes = new_figure.add_subplot(111)
    axes.set_yscale("log", nonposy='clip')
    axes.hist(data)
    axes.set_xlabel('Weight')
    axes.set_ylabel('Instances')
    axes.grid(True)
    plt.show()