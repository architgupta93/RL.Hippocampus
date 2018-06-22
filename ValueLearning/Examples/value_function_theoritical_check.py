# Testing if the value function we learn with a random agent matches what is
# theoritically suggested for a random agent.

import Environment
import Hippocampus
import ValueLearning
import Agents

import matplotlib.pylab as plt

def testMaze(n_trials, dbg_lvl=1):
    ValueLearning.DBG_LVL = dbg_lvl
    # Create a very small maze
    nx = 4
    ny = 5

    n_fields = round(1.0 * (nx+3) * (ny+3))
    n_cells  = n_fields

    maze         = Environment.RandomGoalOpenField(nx, ny)
    place_fields = Hippocampus.setupPlaceFields(maze, n_fields)
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)

    # Learn the value function
    ValueLearning.learnValueFunction(n_trials, maze, place_cells)

    # Evaluate the theoritical value function for a random policy
    ideal_critic = Agents.IdealValueAgent(maze, place_cells)
    optimal_value_function = ideal_critic.getValueFunction()

    plt.imshow(optimal_value_function)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    n_trials = 100
    testMaze(n_trials, dbg_lvl=1)