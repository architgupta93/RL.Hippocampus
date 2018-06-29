import Hippocampus
import Environment
import ValueLearning
import Graphics
import Agents

# Packages for visualization and analysis
import time
import threading
import numpy as np
import matplotlib.pylab as pl

def testMaze(n_training_trials, n_navigation_trials):
    """
    No comments here. Look at single_maze_learning_agent.py for more details!
    """
    ValueLearning.DBG_LVL = 0
    move_distance = 0.29

    nx = 6
    ny = 6

    n_fields = round(1.0 * (nx + 3) * (ny+3))
    Hippocampus.N_CELLS_PER_FIELD = 4
    n_cells  = Hippocampus.N_CELLS_PER_FIELD * n_fields

    n_alternations = 1
    max_nav_steps = 400
    max_train_steps = 4000

    # First Environment: Has its own place cells and place fields
    env_E1          = Environment.RandomGoalOpenField(nx, ny, move_distance)
    canvas_E1       = Graphics.WallMazeCanvas(env_E1)
    place_fields_E1 = Hippocampus.setupPlaceFields(env_E1, n_fields)
    place_cells_E1  = Hippocampus.assignPlaceCells(n_cells, place_fields_E1)

    # Create empty actors and critics
    actor = Agents.Actor(env_E1.getActions(), n_cells)
    critic = Agents.Critic(n_cells)

    # Second Environment: This has a different set (but the same number) of
    # place fields and place cells
    nx = 6
    ny = 6
    lp_wall = Environment.Wall((0,3), (3,3))
    rp_wall = Environment.Wall((4,3), (6,3))
    env_E2          = Environment.MazeWithWalls(nx, ny, [lp_wall, rp_wall], move_distance)
    canvas_E2       = Graphics.WallMazeCanvas(env_E2)
    place_fields_E2 = Hippocampus.setupPlaceFields(env_E2, n_fields)
    place_cells_E2  = Hippocampus.assignPlaceCells(n_cells, place_fields_E2)

    learning_steps_E1 = np.zeros((n_training_trials, 1), dtype=float)
    learning_steps_E2 = np.zeros((n_training_trials, 1), dtype=float)
    for alt in range(n_alternations):
        print('Alternation: %d' % alt)
        # First look at the performance of the agent in the task before it is
        # allowed to learn anything. Then allow learning

        print('Learning Environment B')
        (actor, critic, steps_E2) = ValueLearning.learnValueFunction(n_training_trials, env_E2, place_cells_E2, actor, critic, max_train_steps)
        learning_steps_E2 = steps_E2

        print('Learning Environment A')
        (actor, critic, steps_E1) = ValueLearning.learnValueFunction(n_training_trials, env_E1, place_cells_E1, actor, critic, max_train_steps)
        learning_steps_E1 = steps_E1

    # canvas_E1.plotValueFunction(place_cells_E1, critic)
    # canvas_E2.plotValueFunction(place_cells_E2, critic)

    # Plot a histogram of the weights
    # Critic
    # critic_weights = np.reshape(critic.getWeights(), -1)
    # Graphics.histogram(critic_weights)

    """
    # Actor
    actor_weights = np.reshape(actor.getWeights(), -1)
    Graphics.histogram(actor_weights)
    """

    # After alternation, check the behavior on both the tasks
    n_trials = n_navigation_trials
    ValueLearning.DBG_LVL = 0
    print('Navigating Environment A')
    navigation_steps_E1 = ValueLearning.navigate(n_trials, env_E1, place_cells_E1, actor, critic, max_nav_steps)

    print('Navigating Environment B')
    navigation_steps_E2 = ValueLearning.navigate(n_trials, env_E2, place_cells_E2, actor, critic, max_nav_steps)

    return (learning_steps_E1, learning_steps_E2, navigation_steps_E1, navigation_steps_E2)

class MazeThread(threading.Thread):
    def __init__(self, thread_id, n_train, n_nav):
        threading.Thread.__init__(self)
        self._thread_id = thread_id
        self._n_train = n_train
        self._n_nav = n_nav
        self.training_steps_E1 = None
        self.training_steps_E2 = None
        self.navigation_steps_E1 = None
        self.navigation_steps_E2 = None

    def run(self):
        print("Starting Thread: ", self._thread_id)
        (self.training_steps_E1, self.training_steps_E2, self.navigation_steps_E1, self.navigation_steps_E2)  = testMaze(self._n_train, self._n_nav)
        print("Exiting Thread:", self._thread_id)
        return

if __name__ == "__main__":
    n_epochs = 10
    n_training_trials = 50 # Training trials
    n_navigation_trials = 20  # Navigation trials

    threads = [None] * n_epochs

    for epoch in range(n_epochs):
        threads[epoch] = MazeThread(epoch, n_training_trials, n_navigation_trials)
        threads[epoch].start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    training_steps_E1 = np.zeros((n_training_trials, n_epochs), dtype=float)
    navigation_steps_E1 = np.zeros((n_navigation_trials, n_epochs), dtype=float)
    training_steps_E2 = np.zeros((n_training_trials, n_epochs), dtype=float)
    navigation_steps_E2 = np.zeros((n_navigation_trials, n_epochs), dtype=float)
    for epoch in range(n_epochs):
        training_steps_E1[:, epoch] = threads[epoch].training_steps_E1
        navigation_steps_E1[:, epoch] = threads[epoch].navigation_steps_E1
        training_steps_E2[:, epoch] = threads[epoch].training_steps_E2
        navigation_steps_E2[:, epoch] = threads[epoch].navigation_steps_E2

    mean_training_steps_E1   = np.reshape(np.mean(training_steps_E1, axis=1), (n_training_trials, 1))
    mean_navigation_steps_E1 = np.reshape(np.mean(navigation_steps_E1, axis=1), (n_navigation_trials, 1))
    mean_training_steps_E2   = np.reshape(np.mean(training_steps_E2, axis=1), (n_training_trials, 1))
    mean_navigation_steps_E2 = np.reshape(np.mean(navigation_steps_E2, axis=1), (n_navigation_trials, 1))

    # For plotting the standard deviation, use this!
    err_training_steps_E1 = np.std(training_steps_E1, axis=1)
    err_navigation_steps_E1 = np.std(navigation_steps_E1, axis=1)
    err_training_steps_E2 = np.std(training_steps_E2, axis=1)
    err_navigation_steps_E2 = np.std(navigation_steps_E2, axis=1)

    print('Plotting Statistics for Environment 1')
    training_fig = pl.figure()
    training_ax = training_fig.add_subplot(111)
    training_ax.errorbar(range(n_training_trials), mean_training_steps_E1, yerr=err_training_steps_E1, marker='d', ecolor='black', capsize=0.5)
    training_ax.set_xlabel('Trials')
    training_ax.set_ylabel('Latency')
    training_ax.grid(True)
    pl.gcf().show()

    navigation_fig = pl.figure()
    navigation_ax = navigation_fig.add_subplot(111)
    navigation_ax.errorbar(range(n_navigation_trials), mean_navigation_steps_E1, yerr=err_navigation_steps_E1, marker='o', ecolor='black', capsize=0.5)
    navigation_ax.set_xlabel('Trials')
    navigation_ax.set_ylabel('Latency')
    navigation_ax.grid(True)
    pl.show()

    print('Plotting Statistics for Environment 1')
    training_fig = pl.figure()
    training_ax = training_fig.add_subplot(111)
    training_ax.errorbar(range(n_training_trials), mean_training_steps_E2, yerr=err_training_steps_E2, marker='d', ecolor='black', capsize=0.5)
    training_ax.set_xlabel('Trials')
    training_ax.set_ylabel('Latency')
    training_ax.grid(True)
    pl.gcf().show()

    navigation_fig = pl.figure()
    navigation_ax = navigation_fig.add_subplot(111)
    navigation_ax.errorbar(range(n_navigation_trials), mean_navigation_steps_E2, yerr=err_navigation_steps_E2, marker='o', ecolor='black', capsize=0.5)
    navigation_ax.set_xlabel('Trials')
    navigation_ax.set_ylabel('Latency')
    navigation_ax.grid(True)
    pl.show()

    print('Execution complete. Exiting!')