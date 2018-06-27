# Testing if the value function we learn with a random agent matches what is
# theoritically suggested for a random agent.

import Environment
import Hippocampus
import ValueLearning
import Agents
import Graphics

import numpy as np

def testMaze(n_trials, dbg_lvl=1):
    ValueLearning.DBG_LVL = dbg_lvl

    # Open field - Rather boring
    # maze         = Environment.RandomGoalOpenField(nx, ny)

    # Maze with partition - 6 x 4 environment
    #           ----------------- (6,4)
    #           |               |
    #           | (2,2)   (4,2) |
    #           |-----     -----| (6,2)
    #           |               |
    #           |               |
    #     (0,0) -----------------

    nx = 6
    ny = 6
    maze = Environment.RandomGoalOpenField(nx, ny)

    # Adding walls and constructing the environment
    """
    nx = 6
    ny = 4
    lp_wall = Environment.Wall((0,2), (2,2))
    rp_wall = Environment.Wall((4,2), (6,2))
    maze    = Environment.MazeWithWalls(nx, ny, [lp_wall, rp_wall])
    """

    # Maze with walls - 10 x 10 environment
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

    """
    nx = 10
    ny = 10
    # Adding walls and constructing the environment
    lh_wall = Environment.Wall((2,4), (6,4))
    lv_wall = Environment.Wall((2,2), (2,10))
    rh_wall = Environment.Wall((4,6), (8,6))
    rv_wall = Environment.Wall((8,0), (8,8))
    maze    = Environment.MazeWithWalls(nx, ny, [lh_wall, lv_wall, rh_wall, rv_wall])
    """

    n_fields     = round(1.0 * (nx+3) * (ny+3))
    n_cells      = n_fields
    Hippocampus.N_CELLS_PER_FIELD = 1
    place_fields = Hippocampus.setupPlaceFields(maze, n_fields)
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)

    # Learn the value function
    amateur_critic = None
    n_episodes     = 100
    training_eps   = round(n_episodes/2)
    canvas         = Graphics.MazeCanvas(maze)
    weights        = np.empty((n_cells, n_episodes), dtype=float)
    for episode in range(n_episodes):
        (_, amateur_critic, _) = ValueLearning.learnValueFunction(n_trials, maze, place_cells, critic=amateur_critic, max_steps=1000)
        weights[:, episode]    = amateur_critic.getWeights()
        # canvas.plotValueFunction(place_cells, amateur_critic, continuous=True)
        print('Ended Episode %d'% episode)
        # input()

    # Draw the final value funciton
    canvas.plotValueFunction(place_cells, amateur_critic, continuous=True)

    """ DEBUG
    print(components.explained_variance_ratio_)
    print(components.singular_values_)
    """

    # Graphics.showDecomposition(weights)

    # Evaluate the theoritical value function for a random policy
    ideal_critic = Agents.IdealValueAgent(maze, place_cells)
    optimal_value_function = ideal_critic.getValueFunction()

    scaling_factor = 1.0/(1 - amateur_critic.getDiscountFactor())
    Graphics.showImage(optimal_value_function, range=(maze.NON_GOAL_STATE_REWARD, scaling_factor * maze.GOAL_STATE_REWARD))
    input('Press any key to Exit!')

if __name__ == "__main__":
    n_trials = 1
    testMaze(n_trials, dbg_lvl=0)