import Hippocampus
import Environment
import ValueLearning
import Graphics

# Packages for local plotting/analysis
import time
import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as pl
from pprint import pprint
from scipy import stats

MOVE_DISTACE = 0.29

def testMaze(n_steps, learning_dbg_lvl=0, navigation_dbg_lvl=0):
    nT = n_steps[0] # Training steps
    nN = n_steps[1] # Navigation steps
    ValueLearning.DBG_LVL = learning_dbg_lvl
    # Create a Maze for the experiment
    nx = 6
    ny = 6

    move_distance = MOVE_DISTACE

    # Every location has an associated place field
    # TODO: Play around with having more/fewer place fields!
    n_fields = round(1.0 * (nx+3) * (ny+3))

    # Instead of having multiple cells per field, here we can get away with
    # having fewer place fields than the number of locations on the map (which
    # makes sense which the fields are smeared across the space).
    Hippocampus.N_CELLS_PER_FIELD = 4
    n_cells  = n_fields * Hippocampus.N_CELLS_PER_FIELD

    # Build the maze
    maze  = Environment.RandomGoalOpenField(nx, ny, move_distance)
    canvas = Graphics.WallMazeCanvas(maze)

    # Generate a set of place fields for the environment
    place_fields = Hippocampus.setupPlaceFields(maze, n_fields) 
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)
    if (learning_dbg_lvl > 1):
        canvas.visualizePlaceFields(place_cells)

    # Learn how to navigate this Environment
    (actor, critic, learning_steps) = ValueLearning.learnValueFunction(nT, maze, place_cells, max_steps=4000)
    print(learning_steps)

    # Try a single trial on the same Maze and see how we do
    ValueLearning.DBG_LVL = navigation_dbg_lvl
    navigation_steps = ValueLearning.navigate(nN, maze, place_cells, actor, critic, max_steps=400)
    print(navigation_steps)
    return (learning_steps, navigation_steps)

class MazeThread(threading.Thread):
    def __init__(self, thread_id, n_train, n_nav):
        threading.Thread.__init__(self)
        self._thread_id = thread_id
        self._n_train = n_train
        self._n_nav = n_nav
        self.training_steps = None
        self.navigation_steps = None

    def run(self):
        print("Starting Thread: ", self._thread_id)
        (self.training_steps, self.navigation_steps)  = testMaze(self._n_train, self._n_nav)
        print("Exiting Thread:", self._thread_id)
        return

def main(n_epochs=1):
    if n_epochs == 1:
        # Single trial
        n_training_trials = 40 # Training trials
        n_navigation_trials = 20  # Navigation trials
    elif n_epochs == 3:
        # For reasonable data
        n_training_trials = 100 # Training trials
        n_navigation_trials = 20  # Navigation trials
    else:
        # For quick trials
        n_training_trials = 20 # Training trials
        n_navigation_trials = 20  # Navigation trials

    training_steps = np.zeros((n_training_trials, n_epochs), dtype=float)
    navigation_steps = np.zeros((n_navigation_trials, n_epochs), dtype=float)

    """
    # This version uses threads, which are quite slow in python because of GIL
    threads = [None] * n_epochs
    for epoch in range(n_epochs):
        threads[epoch] = MazeThread(epoch, n_training_trials, n_navigation_trials)
        threads[epoch].start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    for epoch in range(n_epochs):
        training_steps[:, epoch] = threads[epoch].training_steps
        navigation_steps[:, epoch] = threads[epoch].navigation_steps
    """

    threads = multiprocessing.Pool(n_epochs)
    training_results = threads.map(testMaze, [(n_training_trials, n_navigation_trials) for x in range(n_epochs)])
    for epoch in range(n_epochs):
        training_steps[:, epoch] = training_results[epoch][0] * MOVE_DISTACE
        navigation_steps[:, epoch] = training_results[epoch][1] * MOVE_DISTACE

    mean_training_steps   = np.reshape(np.mean(training_steps, axis=1), (n_training_trials, 1))
    mean_navigation_steps = np.reshape(np.mean(navigation_steps, axis=1), (n_navigation_trials, 1))

    # Use this for plotting absolute data deviation
    """
    min_dev_training_steps = np.reshape(np.min(training_steps-mean_training_steps, axis=1), (1, n_training_trials))
    max_dev_training_steps = np.reshape(np.max(training_steps-mean_training_steps, axis=1), (1, n_training_trials))
    err_training_steps = np.abs(np.append(min_dev_training_steps, max_dev_training_steps, axis=0))

    min_dev_navigation_steps = np.reshape(np.min(navigation_steps-mean_navigation_steps, axis=1), (1, n_navigation_trials))
    max_dev_navigation_steps = np.reshape(np.max(navigation_steps-mean_navigation_steps, axis=1), (1, n_navigation_trials))
    err_navigation_steps = np.abs(np.append(min_dev_navigation_steps, max_dev_navigation_steps, axis=0))
    """

    # For plotting the SEM measure, use this!
    err_training_steps = stats.sem(training_steps, axis=1)
    err_navigation_steps = stats.sem(navigation_steps, axis=1)

    # Print all the data before plotting
    # print('%d Training trials'%n_training_trials)
    # pprint(mean_training_steps)

    training_fig = pl.figure()
    training_ax = training_fig.add_subplot(111)
    training_ax.errorbar(range(n_training_trials),
            mean_training_steps,
            yerr=err_training_steps,
            marker='d',
            linewidth=2.0,
            elinewidth=2.0,
            alpha=0.7,
            ecolor='black',
            capsize=2.0)
    training_ax.set_xlabel('Trials')
    training_ax.set_ylabel('Distance Moved')
    training_ax.set_xlim((-1, 20))
    training_ax.set_xticks((0, 5, 10, 15, 20))
    training_ax.set_ylim((0, 100))
    training_ax.set_yticks((0, 50, 100))
    Graphics.cleanAxes(training_ax)
    pl.show()

    # print('%d Navigation trials'%n_navigation_trials)
    # pprint(mean_navigation_steps)

    navigation_fig = pl.figure()
    navigation_ax = navigation_fig.add_subplot(111)
    navigation_ax.errorbar(range(n_navigation_trials),
            mean_navigation_steps,
            yerr=err_navigation_steps,
            marker='o',
            linewidth=2.0,
            elinewidth=2.0,
            ecolor='black',
            alpha=0.7,
            capsize=2.0)
    navigation_ax.set_xlabel('Trials')
    navigation_ax.set_ylabel('Distance Moved')
    navigation_ax.set_ylim((0, 100))
    navigation_ax.set_xlim((-1, 20))
    navigation_ax.set_xticks((0, 5, 10, 15, 20))
    navigation_ax.set_yticks((0, 50, 100))
    Graphics.cleanAxes(navigation_ax)
    pl.show()

    print('Execution complete. Exiting!')

if __name__ == "__main__":
    main(25)
