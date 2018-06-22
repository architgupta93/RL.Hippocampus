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
    nx = 10
    ny = 10

    n_fields = round(1.0 * (nx+3) * (ny+3))
    n_cells  = n_fields

    # Open field - Rather boring
    # maze         = Environment.RandomGoalOpenField(nx, ny)

    # Maze with walls
    #           (2,10)   (8,10)
    #       --------------------- (10,10)
    #       |    |              |
    #       |    |  (4, 6) |    | (10, 8)
    #       |    |   ------|    | 
    #       |    |  (6,4)  |    |
    # (0,4) |    |------   |    |
    #       |    |         |    |
    #       |     (2,2)    |    |
    # (0,0) ---------------------

    # Adding walls and constructing the environment
    lh_wall = Environment.Wall((2,4), (6,4))
    lv_wall = Environment.Wall((2,2), (2,10))
    rh_wall = Environment.Wall((4,6), (8,6))
    rv_wall = Environment.Wall((8,0), (8,8))
    maze    = Environment.MazeWithWalls(nx, ny, [lh_wall, lv_wall, rh_wall, rv_wall])

    place_fields = Hippocampus.setupPlaceFields(maze, n_fields)
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)

    # Learn the value function
    ValueLearning.learnValueFunction(n_trials, maze, place_cells, max_steps=1000)

    # Evaluate the theoritical value function for a random policy
    ideal_critic = Agents.IdealValueAgent(maze, place_cells)
    optimal_value_function = ideal_critic.getValueFunction()

    plt.imshow(optimal_value_function)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    n_trials = 2
    testMaze(n_trials, dbg_lvl=0)