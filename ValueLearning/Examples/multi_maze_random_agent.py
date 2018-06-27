import Hippocampus
import Environment
import ValueLearning
import Graphics
import Agents

# Packages for visualization and analysis
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pylab as plt

def testMaze():
    """
    No comments here. Look at single_maze_learning_agent.py for more details!
    """
    ValueLearning.DBG_LVL = 0

    nx = 6
    ny = 6

    # Set the number of cells to be used per "place field" - Same for all the environments
    Hippocampus.N_CELLS_PER_FIELD = 64

    n_fields = round(1.0 * (nx + 3) * (ny+3))
    n_cells  = Hippocampus.N_CELLS_PER_FIELD * n_fields

    n_training_trials = 1
    n_single_env_episodes = 10
    n_alternations = 10
    max_train_steps = 1000

    # First Environment: Has its own place cells and place fields
    env_E1          = Environment.RandomGoalOpenField(nx, ny)
    place_fields_E1 = Hippocampus.setupPlaceFields(env_E1, n_fields)
    place_cells_E1  = Hippocampus.assignPlaceCells(n_cells, place_fields_E1)

    # Train a critic on the first environment
    print('Training Critic solely on Env A')
    critic_E1  = None
    weights_E1 = np.empty((n_single_env_episodes, n_cells), dtype=float) 
    for episode in range(n_single_env_episodes):
        (_, critic_E1, _) = ValueLearning.learnValueFunction(n_training_trials, env_E1, place_cells_E1, critic=critic_E1, max_steps=max_train_steps)
        weights_E1[episode, :] = critic_E1.getWeights()

    components_E1 = Graphics.showDecomposition(weights_E1)

    # Create empty actors and critics
    actor = Agents.RandomAgent(env_E1.getActions(), n_cells)
    critic = Agents.Critic(n_cells)

    # Second Environment: This has a different set (but the same number) of
    # place fields and place cells
    env_E2          = Environment.RandomGoalOpenField(nx, ny)
    place_fields_E2 = Hippocampus.setupPlaceFields(env_E2, n_fields)
    place_cells_E2  = Hippocampus.assignPlaceCells(n_cells, place_fields_E2)

    # Train another critic on the second environment
    print()
    print('Training Critic solely on Env B')
    critic_E2  = None
    weights_E2 = np.empty((n_single_env_episodes, n_cells), dtype=float) 
    for episode in range(n_single_env_episodes):
        (_, critic_E2, _) = ValueLearning.learnValueFunction(n_training_trials, env_E2, place_cells_E2, critic=critic_E2, max_steps=max_train_steps)
        weights_E2[episode, :] = critic_E2.getWeights()

    components_E2 = Graphics.showDecomposition(weights_E2)

    # Look at the projection of one environment's weights on the other's principal components
    Graphics.showDecomposition(weights_E1, components=components_E2, title='E2 on E1')
    Graphics.showDecomposition(weights_E2, components=components_E1, title='E1 on E2')
    input('Press any key to start Alternation.')

    # This can be used to just reinforce the fact that the agent is indeed
    # random! The steps taken to goal would not change over time because of the
    # way the agent behaves.
    learning_steps_E1 = np.zeros((n_alternations, 1), dtype=float)
    learning_steps_E2 = np.zeros((n_alternations, 1), dtype=float)

    # keep track of weights for PCA
    weights        = np.empty((n_alternations * 2, n_cells), dtype=float)
    for alt in range(n_alternations):
        print('Alternation: %d' % alt)
        # First look at the performance of the agent in the task before it is
        # allowed to learn anything. Then allow learning
        print('Learning Environment A')
        (actor, critic, steps_E1) = ValueLearning.learnValueFunction(n_training_trials, env_E1, place_cells_E1, actor, critic, max_train_steps)
        learning_steps_E1[alt] = np.mean(steps_E1)
        weights[2*alt, :] = critic.getWeights()

        # Repeat for environment 1
        print('Learning Environment B')
        (actor, critic, steps_E2) = ValueLearning.learnValueFunction(n_training_trials, env_E2, place_cells_E2, actor, critic, max_train_steps)
        learning_steps_E2[alt] = np.mean(steps_E2)
        weights[2*alt + 1, :] = critic.getWeights()

    # Show the alternation weights in the two basis
    Graphics.showDecomposition(weights, components=components_E1, title='Alternation weights in E1')
    Graphics.showDecomposition(weights, components=components_E2, title='Alternation weights in E2')
    input('Press any key to exit!')
    
    # joint_components = Graphics.showDecomposition(weights)

    """ DEBUG
    print(components.explained_variance_ratio_)
    print(components.singular_values_)
    """

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