import Hippocampus
import Environment
import ValueLearning
import Graphics

# Packages for local plotting/analysis
import time
import threading
import numpy as np
import matplotlib.pyplot as pl

def testMaze(nT, nN, learning_dbg_lvl=0, navigation_dbg_lvl=0):
    ValueLearning.DBG_LVL = learning_dbg_lvl
    # Create a Maze for the experiment
    nx = 10
    ny = 10

    # Every location has an associated place field
    # TODO: Play around with having more/fewer place fields!
    nf = round(0.5 * (nx * ny))

    # Build the maze
    maze  = Environment.RandomGoalOpenField(nx, ny)
    canvas = Graphics.MazeCanvas(maze)

    # Generate a set of place fields for the environment
    place_fields = Hippocampus.setupPlaceFields(maze, nf) 
    if (learning_dbg_lvl > 0):
        canvas.visualizePlaceFields(place_fields)

    # Learn how to navigate this Environment
    (actor, critic, learning_steps) = ValueLearning.learnValueFunction(nT, maze, place_fields, max_steps=2000)

    # Try a single trial on the same Maze and see how we do
    ValueLearning.DBG_LVL = navigation_dbg_lvl
    navigation_steps = ValueLearning.navigate(nN, maze, place_fields, actor, critic, max_steps=200)
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

if __name__ == "__main__":
    n_epochs = 40
    n_training_trials = 40 # Training trials
    n_navigation_trials = 20  # Navigation trials

    threads = [None] * n_epochs

    training_steps = np.zeros((n_training_trials, n_epochs), dtype=float)
    navigation_steps = np.zeros((n_navigation_trials, n_epochs), dtype=float)
    for epoch in range(n_epochs):
        threads[epoch] = MazeThread(epoch, n_training_trials, n_navigation_trials)
        threads[epoch].start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    for epoch in range(n_epochs):
        training_steps[:, epoch] = threads[epoch].training_steps
        navigation_steps[:, epoch] = threads[epoch].navigation_steps

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

    # For plotting the standard deviation, use this!
    err_training_steps = np.std(training_steps, axis=1)
    err_navigation_steps = np.std(navigation_steps, axis=1)

    training_fig = pl.figure(0)
    training_ax = training_fig.add_subplot(111)
    training_ax.errorbar(range(n_training_trials), mean_training_steps, yerr=err_training_steps, marker='d', ecolor='black', capsize=0.5)

    navigation_fig = pl.figure(1)
    navigation_ax = navigation_fig.add_subplot(111)
    navigation_ax.errorbar(range(n_navigation_trials), mean_navigation_steps, yerr=err_navigation_steps, marker='o', ecolor='black', capsize=0.5)
    pl.show()

    print('Execution complete. Exiting!')