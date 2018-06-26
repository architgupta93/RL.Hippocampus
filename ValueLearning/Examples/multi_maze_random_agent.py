import Hippocampus
import Environment
import ValueLearning
import Graphics
import Agents

# Packages for visualization and analysis
import numpy as np

def testMaze():
    """
    No comments here. Look at single_maze_learning_agent.py for more details!
    """
    ValueLearning.DBG_LVL = 1

    nx = 4
    ny = 5

    # Set the number of cells to be used per "place field" - Same for all the environments
    Hippocampus.N_CELLS_PER_FIELD = 1

    n_fields = round(1.0 * (nx + 3) * (ny+3))
    n_cells  = Hippocampus.N_CELLS_PER_FIELD * n_fields

    n_training_trials = 200
    n_alternations = 4
    max_train_steps = 200

    # First Environment: Has its own place cells and place fields
    env_E1          = Environment.RandomGoalOpenField(nx, ny)
    place_fields_E1 = Hippocampus.setupPlaceFields(env_E1, n_fields)
    place_cells_E1  = Hippocampus.assignPlaceCells(n_cells, place_fields_E1)

    # Create empty actors and critics
    actor = Agents.RandomAgent(env_E1.getActions(), n_cells)
    critic = Agents.Critic(n_cells)

    # Second Environment: This has a different set (but the same number) of
    # place fields and place cells
    env_E2          = Environment.RandomGoalOpenField(nx, ny)
    place_fields_E2 = Hippocampus.setupPlaceFields(env_E2, n_fields)
    place_cells_E2  = Hippocampus.assignPlaceCells(n_cells, place_fields_E2)

    # This can be used to just reinforce the fact that the agent is indeed
    # random! The steps taken to goal would not change over time because of the
    # way the agent behaves.
    learning_steps_E1 = np.zeros((n_alternations, 1), dtype=float)
    learning_steps_E2 = np.zeros((n_alternations, 1), dtype=float)
    for alt in range(n_alternations):
        print('Alternation: %d' % alt)
        # First look at the performance of the agent in the task before it is
        # allowed to learn anything. Then allow learning

        print('Learning Environment A')
        (actor, critic, steps_E1) = ValueLearning.learnValueFunction(n_training_trials, env_E1, place_cells_E1, actor, critic, max_train_steps)
        learning_steps_E1[alt] = np.mean(steps_E1)

        # Repeat for environment 1
        print('Learning Environment B')
        (actor, critic, steps_E2) = ValueLearning.learnValueFunction(n_training_trials, env_E2, place_cells_E2, actor, critic, max_train_steps)
        learning_steps_E2[alt] = np.mean(steps_E2)

    # canvas_E1.plotValueFunction(place_cells_E1, critic)
    # canvas_E2.plotValueFunction(place_cells_E2, critic)

    # Plot a histogram of the weights
    # Critic
    # critic_weights = np.reshape(critic.getWeights(), -1)
    # Graphics.histogram(critic_weights)

    """
    # Look at how the steps taken during learning varied
    Graphics.plot(learning_steps_E1)
    Graphics.plot(learning_steps_E2)
    """

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')