"""
Testing script for evaluating the value function under a random policy
"""
import ValueLearning
import Environment
import Hippocampus
import Graphics
import Agents

import threading

def testMaze(n_train, n_nav):
    ValueLearning.DBG_LVL = 1

    # Experiment parameters
    nx = 5
    ny = 2
    n_fields = round(1.0 * nx * ny)
    n_cells  = n_fields

    # Maze creation
    maze    = Environment.RandomGoalOpenField(nx, ny)
    canvas  = Graphics.MazeCanvas(maze)

    # Generate place fields and place cells
    place_fields = Hippocampus.setupPlaceFields(maze, n_fields)
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)
    canvas.visualizePlaceFields(place_cells)

    # Create Actor and Critic
    actor  = Agents.RandomAgent(maze.getActions(), n_cells)
    critic = Agents.Critic(n_fields)

    ValueLearning.learnValueFunction(n_train, maze, place_cells, actor, critic, max_steps=1000)
    # canvas.plotValueFunction(place_cells, critic)

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
        testMaze(self._n_train, self._n_nav)
        print("Exiting Thread:", self._thread_id)

if __name__ == "__main__":
    n_epochs = 1
    n_train  = 100
    n_nav    = 10
    threads  = [None] * n_epochs

    for epoch in range(n_epochs):
        threads[epoch] = MazeThread(epoch, n_train, n_nav)
        threads[epoch].start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # TODO: Show a plot of the value function and compare with the ideal valuE