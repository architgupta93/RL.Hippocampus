import Hippocampus
import Environment
import ValueLearning
import Graphics

# Plotting and utilitiy (Multi-threading)
import time
import threading
import numpy as np
import matplotlib.pylab as pl

def testMaze(nT, nN, learning_dbg_lvl=2, navigation_dbg_lvl=0):
    ValueLearning.DBG_LVL = learning_dbg_lvl

    # Parameters describing the maze
    nx = 10
    ny = 10

    # Build the Maze (add walls etc.)
    # Design of the task
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

    # Adding two walls (horizontal and vertical)
    h_wall = Environment.Wall((2,4), (4,4))
    v_wall = Environment.Wall((2,2), (2,10))
    maze   = Environment.MazeWithWalls(nx, ny, [h_wall, v_wall])

    # Create a initial and goal location including the information about wall locations

    # Object for plotting and visualization
    canvas = Graphics.WallMazeCanvas(maze)

    # Add Place fields and place cells
    n_fields = round(1.0 * nx * ny)
    n_cells  = n_fields

    place_fields = Hippocampus.setupPlaceFields(maze, n_fields)
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)

    if (learning_dbg_lvl > 1):
        canvas.visualizePlaceFields(place_cells)

    # Learn how to navigate this Environment
    (actor, critic, learning_steps) = ValueLearning.learnValueFunction(nT, maze, place_cells, max_steps=1000)

    # Try a single trial on the same Maze and see how we do
    ValueLearning.DBG_LVL = navigation_dbg_lvl
    navigation_steps = ValueLearning.navigate(nN, maze, place_cells, actor, critic, max_steps=100)
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

    print('Execution complete. Exiting!')