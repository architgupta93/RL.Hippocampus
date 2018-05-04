import Hippocampus
import Environment
import ValueLearning
import Graphics
import Agents

# Packages for visualization and analysis
import numpy as np

def testMaze():
    """
    No comments here. Look at single_maze_test.py for more details!
    """
    ValueLearning.DBG_LVL = 1

    nx = 10
    ny = 10

    n_fields = round(0.5 * nx * ny)
    n_cells  = Hippocampus.N_CELLS_PER_FIELD * n_fields

    n_training_trials = 5
    n_navigation_trials = 10

    n_alternations = 10
    max_nav_steps = 100
    max_train_steps = 20

    # First Environment: Has its own place cells and place fields
    env_E1          = Environment.RandomGoalOpenField(nx, ny)
    place_fields_E1 = Hippocampus.setupPlaceFields(env_E1, n_fields)
    place_cells_E1  = Hippocampus.assignPlaceCells(n_cells, place_fields_E1)
    canvas_E1       = Graphics.MazeCanvas(env_E1)

    # canvas_E1.visualizePlaceFields(place_cells_E1)

    # Create empty actors and critics
    actor = Agents.Actor(env_E1.getActions(), n_cells)
    critic = Agents.Critic(n_cells)

    # Second Environment: This has a different set (but the same number) of
    # place fields and place cells
    env_E2          = Environment.RandomGoalOpenField(nx, ny)
    place_fields_E2 = Hippocampus.setupPlaceFields(env_E2, n_fields)
    place_cells_E2  = Hippocampus.assignPlaceCells(n_cells, place_fields_E2)
    canvas_E2       = Graphics.MazeCanvas(env_E2)

    learning_steps_E1 = np.zeros((n_alternations, 1), dtype=float)
    learning_steps_E2 = np.zeros((n_alternations, 1), dtype=float)
    for alt in range(n_alternations):
        print('Alternation: %d' % alt)
        # First look at the performance of the agent in the task before it is
        # allowed to learn anything. Then allow learning

        # print('Navigation Environment B')
        # ValueLearning.navigate(n_navigation_trials, env_E2, place_cells_E2, actor, critic, max_nav_steps)
        print('Learning Environment A')
        (actor, critic, steps_E1) = ValueLearning.learnValueFunction(n_training_trials, env_E1, place_cells_E1, actor, critic, max_train_steps)
        learning_steps_E1[alt] = np.mean(steps_E1)

        # Repeat for environment 1
        # print('Navigation Environment A')
        # ValueLearning.navigate(n_navigation_trials, env_E1, place_cells_E1, actor, critic, max_nav_steps)
        print('Learning Environment B')
        (actor, critic, steps_E2) = ValueLearning.learnValueFunction(n_training_trials, env_E2, place_cells_E2, actor, critic, max_train_steps)
        learning_steps_E2[alt] = np.mean(steps_E2)

    canvas_E1.plotValueFunction(place_cells_E1, critic)
    canvas_E2.plotValueFunction(place_cells_E2, critic)

    # Plot a histogram of the weightS
    critic_weights = np.reshape(critic.getWeights(), -1)
    Graphics.histogram(critic_weights)

    # Look at how the steps taken during learning varied
    Graphics.plot(learning_steps_E1)
    Graphics.plot(learning_steps_E2)

    # After alternation, check the behavior on both the tasks
    n_trials = n_navigation_trials
    ValueLearning.DBG_LVL = 1
    print('Navigating Environment A')
    ValueLearning.navigate(n_trials, env_E1, place_cells_E1, actor, critic, max_nav_steps)

    print('Navigating Environment B')
    ValueLearning.navigate(n_trials, env_E2, place_cells_E2, actor, critic, max_nav_steps)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')