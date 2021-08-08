import Hippocampus
import Environment
import ValueLearning
import Graphics

# Plotting and utilitiy (Multi-threading)
import time
import threading
import multiprocessing
import numpy as np
from scipy import stats
import matplotlib.pylab as pl
MOVE_DISTANCE = 0.03

def testMaze(n_steps, learning_dbg_lvl=1, navigation_dbg_lvl=0):
    nT = n_steps[0] # Training steps
    nN = n_steps[1] # Navigation steps
    ValueLearning.DBG_LVL = learning_dbg_lvl
    move_distance = MOVE_DISTANCE

    # Parameters describing the maze
    # Build the Maze (add walls etc.)
    # Maze with partition - 6 x 6 environment
    #           ----------------- (6,6)
    #           |               |
    #           | (2,3)   (4,3) |
    #           |-----     -----| (6,3)
    #           |               |
    #           |               |
    #     (0,0) -----------------

    nx = 6
    ny = 6
    lp_wall = Environment.Wall((0,3), (2,3))
    rp_wall = Environment.Wall((4,3), (6,3))
    maze    = Environment.MazeWithWalls(nx, ny, [lp_wall, rp_wall], move_distance)

    # 10 x 10 Maze with an L barrier
    #           (2,10)
    #       --------------------- (10,10)
    #       |    |              |
    #       |    |              |
    #       |    |              |
    #       |    |     (4,4)    |
    # (0,4) |    |------        |
    #       |    |              |
    #       |     (2,2)         |
    # (0,0) ---------------------

    """
    nx = 10
    ny = 10

    h_wall = Environment.Wall((2,4), (4,4))
    v_wall = Environment.Wall((2,2), (2,10))
    maze   = Environment.MazeWithWalls(nx, ny, [h_wall, v_wall])
    """

    # Create a initial and goal location including the information about wall locations

    # Object for plotting and visualization
    canvas = Graphics.WallMazeCanvas(maze)

    # Add Place fields and place cells
    n_fields = round(1.0 * (nx + 3) * (ny + 3))
    Hippocampus.N_CELLS_PER_FIELD = 4
    n_cells  = n_fields * Hippocampus.N_CELLS_PER_FIELD

    place_fields = Hippocampus.setupPlaceFields(maze, n_fields)
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)

    if (learning_dbg_lvl > 2):
        canvas.visualizePlaceFields(place_cells)

    # Learn how to navigate this Environment
    (actor, critic, learning_steps) = ValueLearning.learnValueFunction(nT, maze, place_cells, max_steps=4000)

    # Try a single trial on the same Maze and see how we do
    ValueLearning.DBG_LVL = navigation_dbg_lvl
    navigation_steps = ValueLearning.navigate(nN, maze, place_cells, actor, critic, max_steps=400)
    return (learning_steps, navigation_steps)

class MazeThread(threading.Thread):
    def __init__(self, thread_id, n_train, n_nav):
        threading.Thread.__init__(self)
        self._thread_id = thread_id
        self._n_train = n_train
        self._n_nav = n_nav
        self.training_steps = None
        self.navigation_steps = None

        return

    def run(self):
        print("Starting Thread: ", self._thread_id)
        (self.training_steps, self.navigation_steps)  = testMaze(self._n_train, self._n_nav)
        print("Exiting Thread:", self._thread_id)

if __name__ == "__main__":
    n_epochs = 1
    n_training_trials = 100 # Training trials
    n_navigation_trials = 20  # Navigation trials

    """
    threads = [None] * n_epochs

    training_steps = np.zeros((n_training_trials, n_epochs), dtype=float)
    navigation_steps = np.zeros((n_navigation_trials, n_epochs), dtype=float)
    for epoch in range(n_epochs):
        threads[epoch] = MazeThread(epoch, n_training_trials, n_navigation_trials)
        threads[epoch].start()

    # Wait for all threads to complete
    for t in threads:
        t.join()
    """

    training_steps = np.zeros((n_training_trials, n_epochs), dtype=float)
    navigation_steps = np.zeros((n_navigation_trials, n_epochs), dtype=float)

    threads = multiprocessing.Pool(n_epochs)
    training_results = threads.map(testMaze, [(n_training_trials, n_navigation_trials) for x in range(n_epochs)])
    for epoch in range(n_epochs):
        training_steps[:, epoch] = training_results[epoch][0] * MOVE_DISTANCE
        navigation_steps[:, epoch] = training_results[epoch][1] * MOVE_DISTANCE

    mean_training_steps   = np.reshape(np.mean(training_steps, axis=1), (n_training_trials, 1))
    mean_navigation_steps = np.reshape(np.mean(navigation_steps, axis=1), (n_navigation_trials, 1))

    # For plotting the SEM measure, use this!
    err_training_steps = stats.sem(training_steps, axis=1)
    err_navigation_steps = stats.sem(navigation_steps, axis=1)

    training_fig = pl.figure()
    training_ax = training_fig.add_subplot(111)
    training_ax.errorbar(range(n_training_trials), mean_training_steps, yerr=err_training_steps, marker='d', ecolor='black', capsize=0.5)
    training_ax.set_xlabel('Trials')
    training_ax.set_ylabel('Distance Moved')
    Graphics.cleanAxes(training_ax)
    pl.show()

    navigation_fig = pl.figure()
    navigation_ax = navigation_fig.add_subplot(111)
    navigation_ax.errorbar(range(n_navigation_trials), mean_navigation_steps, yerr=err_navigation_steps, marker='o', ecolor='black', capsize=0.5)
    navigation_ax.set_xlabel('Trials')
    navigation_ax.set_ylabel('Distance Moved')
    Graphics.cleanAxes(navigation_ax)
    pl.show()

    print('Execution complete. Exiting!')
