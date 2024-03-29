import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from MotionAnimation.PY import data_types as GR
from sklearn.decomposition import PCA
import numpy as np

MAX_TICKS_TO_SHOW = 5
N_PLACE_CELLS_TO_SHOW = 5

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
AXES_LINE_THICCCK = 2.0

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

def cleanAxes(ax_obj):
    ax_obj.spines['top'].set_visible(False)
    ax_obj.spines['right'].set_visible(False)
    ax_obj.spines['bottom'].set_linewidth(AXES_LINE_THICCCK)
    ax_obj.spines['left'].set_linewidth(AXES_LINE_THICCCK)
    ax_obj.tick_params(axis="y",direction="in", left="off",labelleft="on") 
    ax_obj.tick_params(axis="x",direction="in", bottom="off",labelbottom="on") 
    ax_obj.xaxis.set_tick_params(width=AXES_LINE_THICCCK)
    ax_obj.yaxis.set_tick_params(width=AXES_LINE_THICCCK)
    ax_obj.grid(False)
    plt.tight_layout()

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

        self._v_min    = maze.NON_GOAL_STATE_REWARD
        self._v_max    = maze.GOAL_STATE_REWARD 

    def visualizePlaceField(self, place_cell):
        """
        Show place cell activity for all the positions on the maze for a
        SINGLE cell.
        """
        x_locs = np.linspace(self._min_x, self._max_x, 200)
        y_locs = np.linspace(self._min_y, self._max_y, 200)
        activity = np.zeros((len(x_locs), len(y_locs)), dtype=float)
        for xi, px in enumerate(x_locs):
            for yj, py in enumerate(y_locs):
                activity[xi, yj] = place_cell.getActivity((px, py))

        showImage(activity, xticks=x_locs, yticks=y_locs)

    def visualizePlaceFields(self, cell_list):
        # Select a random subsample of cells to show the place fields for.
        n_cells = len(cell_list)
        cells_to_show = np.random.permutation(n_cells)[:N_PLACE_CELLS_TO_SHOW]

        for cell in cells_to_show:
            self.visualizePlaceField(cell_list[cell])

    def visualizeAggregatePlaceFields(self, place_cells):
        """
        Show place cell activity for all the positions on the maze aggregated
        over all cells.
        """

        x_locs = np.linspace(self._min_x, self._max_x, 200)
        y_locs = np.linspace(self._min_y, self._max_y, 200)
        activity = np.zeros((len(x_locs), len(y_locs)), dtype=float)
        for xi, px in enumerate(x_locs):
            for yj, py in enumerate(y_locs):
                activity[xi, yj] = sum([pf.getActivity((px, py)) for pf in place_cells])

        showImage(activity, xticks=x_locs, yticks=y_locs)

    def update(self, next_state):
        # Append the data points to the appropriate trajectory object
        self._anim_obj.update(self._t_stamp, next_state[0], next_state[1])
        self._t_stamp += 1

    def plotTrajectory(self, fig=None):
        self._anim_obj.setLims((self._min_x, self._max_x), (self._min_y, self._max_y))
        self._anim_obj.plotStaticTR(object_type='line', figure_handle=fig)
    
    def animateTrajectory(self):
        self._anim_obj.plotTimedTR(object_type='point')
    
    def plotValueFunction(self, place_cells, critic, continuous=False, limits=True):
        if continuous:
            # Scale the number of data points in each dimension independently
            nx_pts = round(20 * (self._max_x - self._min_x - 2))
            ny_pts = round(20 * (self._max_y - self._min_y - 2))
            x_locs = np.linspace(self._min_x+1, self._max_x-1, num=nx_pts)
            y_locs = np.linspace(self._min_y+1, self._max_y-1, num=ny_pts)
        else:
            x_locs = range(self._min_x+1, self._max_x)
            y_locs = range(self._min_y+1, self._max_y)

        # Have never quite understood why Meshgrid works in such a wierd way
        values = np.zeros((len(x_locs), len(y_locs)), dtype=float)
        for xi, px in enumerate(x_locs):
            for yj, py in enumerate(y_locs):
                pf_activity = [pf.getActivity((px, py)) for pf in place_cells]
                values[xi, yj] = critic.getValue(pf_activity)
        
        if continuous:
            showSurface(values, xticks=x_locs, yticks=y_locs)

        scaling_factor = 1.0/(1-critic.getDiscountFactor())
        if limits:
            showImage(values, xticks=x_locs, yticks=y_locs, range=(self._v_min, scaling_factor * self._v_max))
        else:
            # Let the image be colored according to its own scale
            showImage(values, xticks=x_locs, yticks=y_locs)

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

        # Show the goal location
        goal_loc = self._maze.getGoalLocation()
        # fig.plot(goal_loc[0], goal_loc[1], color='g', marker='s')
        fig.plot(goal_loc[0], goal_loc[1], 'g-s')

        for idx, wall in enumerate(self._maze.getWalls()):
            [xs, ys] = wall.getPlottingData()
            if idx < 4:
                # These should be the boundaries - Drawn transparent black
                # fig.plot(xs, ys, color='black', lw=3.0, alpha=0.9)
                fig.plot(xs, ys, 'black')
            else:
                # All obstructions (user designed) - Drawn transparent black
                # fig.plot(xs, ys, color='black', lw=3.0, alpha=0.9)
                fig.plot(xs, ys, 'black')

        super().plotTrajectory(fig)

def plot(*args):
    # Create a new figure
    new_figure = plt.figure()
    axes = new_figure.add_subplot(111)
    axes.plot(*args)
    # axes.set_xlabel('Trial')
    # axes.set_ylabel('Latency')
    plt.grid()
    plt.gcf().show()

def histogram(data):
    new_figure = plt.figure()
    axes = new_figure.add_subplot(111)
    axes.set_yscale("log", nonposy='clip')
    axes.hist(data)
    axes.set_xlabel('Bin')
    axes.set_ylabel('Instances')
    axes.grid(True)
    plt.gcf().show()

def showImage(data, xticks=None, yticks=None, range=None, title=None):
    """
    Used to show 2D data using imshow from matplotlib. Issues regarding
    transposition of data, and setting up the correct origin are resolved
    """

    plt.figure()
    data_shape = np.shape(data)
    plt.imshow(data.T, origin='lower')
    # NOTE: LINSPACE in numpy includes the last point, whereas the vanilla
    # 'range' does not. This is the cause of all the pain below!
    if xticks is not None:
        if len(xticks) > MAX_TICKS_TO_SHOW:
            plt.xticks(np.linspace(0, data_shape[0]-1, num=MAX_TICKS_TO_SHOW), np.round(np.linspace(xticks[0], xticks[-1], num=MAX_TICKS_TO_SHOW), 2))
        else:
            plt.xticks(np.linspace(0, data_shape[0]-1, num=len(xticks)), np.round(xticks, 2))

    if yticks is not None:
        if len(yticks) > MAX_TICKS_TO_SHOW:
            plt.yticks(np.linspace(0, data_shape[1]-1, num=MAX_TICKS_TO_SHOW), np.round(np.linspace(yticks[0], yticks[-1], num=MAX_TICKS_TO_SHOW), 2))
        else:
            plt.yticks(np.linspace(0, data_shape[1]-1, num=len(yticks)), np.round(yticks, 2))

    if range is not None:
        plt.clim(vmin=range[0], vmax=range[1])

    if title is not None:
        plt.title(title)

    plt.xlabel('X (bin)')
    plt.ylabel('Y (bin)')
    plt.colorbar()
    plt.gcf().show()

def showSurface(data, xticks=None, yticks=None):
    """
    Show 2D data as a surface plot. A colorbar plot is sometimes insufficient
    for looking at the values of a function (especially for value function in
    a task where adjoioning states tend to have similar values)
    """
    data_shape = np.shape(data)
    if xticks is None:
        xticks = range(data_shape[0])

    if yticks is not None:
        yticks = range(data_shape[1])

    Y, X = np.meshgrid(yticks, xticks)

    # 3D Figure, needs some effort
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Set the format for the axes ticks
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gcf().show()

def showDecomposition(values, components=None, title=None):
    """
    Show the decomposition of values into principal basis vectors (obtained
    through sigular value decomposition). The component axes can be passed
    in, or new ones can be created from values themselves if nothing is
    passed in.
    """
    MAX_SVS_TO_SHOW = 10

    if components is None:
        # Decompose the weight sequence into its principal components
        components = np.linalg.svd(values, full_matrices=False)

        # Get the corresponding singular values
        singular_values = components[1]
        n_singular_values = len(singular_values)
        svs_to_show = min(MAX_SVS_TO_SHOW, n_singular_values)
        # print(singular_values)

        # Plot the singular values
        plt.figure()
        plt.scatter(range(1, 1+svs_to_show), singular_values[:svs_to_show], marker='s', c='red', alpha=0.8)
        plt.yscale('log')
        plt.ylabel('Singular Value')
        plt.grid()
        plt.gcf().show()
    else:
        singular_values = components[1]

    # Components
    major_component = components[0][:,0]/np.linalg.norm(components[0][:,0])
    minor_component = components[0][:,1]/np.linalg.norm(components[0][:,1])
    vector_norms    = np.linalg.norm(values, axis=0)

    # Get the decomposition of the weight vectors into the constituents 
    n_samples = np.shape(values)[1]
    transformed_values = [np.dot(major_component, values)/vector_norms, np.dot(minor_component, values)/vector_norms]
    plt.figure()
    plt.scatter(transformed_values[0], transformed_values[1], c=range(n_samples), \
        cmap='viridis', marker='d', alpha=0.9)
    plt.xlabel('Major, SV - %.2f'% singular_values[0])
    plt.ylabel('Minor, SV - %.2f'% singular_values[1])
    plt.colorbar()
    plt.grid()

    if title is not None:
        plt.title(title)

    plt.gcf().show()
    return components

