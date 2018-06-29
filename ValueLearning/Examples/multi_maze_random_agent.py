#/bin/python

# Packages for visualization and analysis
import numpy as np

# Local packages
import Hippocampus
import Environment
import ValueLearning
import Agents
import Graphics

def testMaze():
    """
    No comments here. Look at single_maze_learning_agent.py for more details!
    """
    ValueLearning.DBG_LVL = 0

    nx = 6
    ny = 6

    # Set the number of cells to be used per "place field" - Same for all the environments
    Hippocampus.N_CELLS_PER_FIELD = 1

    n_fields = round(1.0 * (nx + 3) * (ny+3))
    n_cells  = Hippocampus.N_CELLS_PER_FIELD * n_fields
    move_distance = 0.99

    n_training_trials = 100
    n_single_env_episodes = 2
    n_alternations = 1
    max_train_steps = 1000

    # First Environment: Has its own place cells and place fields
    env_E1          = Environment.RandomGoalOpenField(nx, ny, move_distance)
    canvas_E1       = Graphics.WallMazeCanvas(env_E1)
    place_fields_E1 = Hippocampus.setupPlaceFields(env_E1, n_fields)
    place_cells_E1  = Hippocampus.assignPlaceCells(n_cells, place_fields_E1)

    # Train a critic on the first environment
    print('Training Critic solely on Env A')
    critic_E1  = None
    weights_E1 = np.empty((n_cells, n_single_env_episodes), dtype=float) 
    for episode in range(n_single_env_episodes):
        (_, critic_E1, _) = ValueLearning.learnValueFunction(n_training_trials, env_E1, place_cells_E1, critic=critic_E1, max_steps=max_train_steps)
        weights_E1[:, episode] = critic_E1.getWeights()

    # Get a trajectory in the environment and plot the value function
    canvas_E1.plotValueFunction(place_cells_E1, critic_E1, continuous=True)
    input('Press return to run next environment...')

    components_E1 = Graphics.showDecomposition(weights_E1, title='Environment 01')

    # Create empty actors and critics
    actor = Agents.RandomAgent(env_E1.getActions(), n_cells)
    critic = Agents.Critic(n_cells)

    # Second Environment: This has a different set (but the same number) of
    # place fields and place cells (also has a bunch of walls)
    nx = 6
    ny = 6
    lp_wall = Environment.Wall((0,3), (3,3))
    rp_wall = Environment.Wall((4,3), (6,3))
    env_E2          = Environment.MazeWithWalls(nx, ny, [lp_wall, rp_wall], move_distance=move_distance)
    canvas_E2       = Graphics.WallMazeCanvas(env_E2)
    place_fields_E2 = Hippocampus.setupPlaceFields(env_E2, n_fields)
    place_cells_E2  = Hippocampus.assignPlaceCells(n_cells, place_fields_E2)

    # Train another critic on the second environment
    print()
    print('Training Critic solely on Env B')
    critic_E2  = None
    weights_E2 = np.empty((n_cells, n_single_env_episodes), dtype=float) 
    for episode in range(n_single_env_episodes):
        (_, critic_E2, _) = ValueLearning.learnValueFunction(n_training_trials, env_E2, place_cells_E2, critic=critic_E2, max_steps=max_train_steps)
        weights_E2[:, episode] = critic_E2.getWeights()

    components_E2 = Graphics.showDecomposition(weights_E2, title='Environment 02')
    canvas_E2.plotValueFunction(place_cells_E2, critic_E2, continuous=True)

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
    weights        = np.empty((n_cells, n_alternations * 2), dtype=float)
    for alt in range(n_alternations):
        n_alternation_trials = n_single_env_episodes * n_training_trials
        # n_alternation_trials = n_training_trials
        print('Alternation: %d' % alt)
        # First look at the performance of the agent in the task before it is
        # allowed to learn anything. Then allow learning
        print('Learning Environment A')
        (actor, critic, steps_E1) = ValueLearning.learnValueFunction(n_alternation_trials, env_E1, place_cells_E1, actor, critic, max_train_steps)
        learning_steps_E1[alt] = np.mean(steps_E1)
        weights[:, 2*alt] = critic.getWeights()

        # Repeat for environment 1
        print('Learning Environment B')
        (actor, critic, steps_E2) = ValueLearning.learnValueFunction(n_alternation_trials, env_E2, place_cells_E2, actor, critic, max_train_steps)
        learning_steps_E2[alt] = np.mean(steps_E2)
        weights[:, 2*alt + 1] = critic.getWeights()

    # Show the alternation weights in the two basis
    Graphics.showDecomposition(weights, components=components_E1, title='Alternation weights in E1')
    Graphics.showDecomposition(weights, components=components_E2, title='Alternation weights in E2')

    # Show the value functions for both the environments
    input('Press return for Value Function of E1')
    canvas_E1.plotValueFunction(place_cells_E1, critic, continuous=True)
    canvas_E1.plotValueFunction(place_cells_E1, critic_E1, continuous=True)
    canvas_E1.plotValueFunction(place_cells_E1, critic_E2, continuous=True)

    # Plot the ideal value function
    ideal_critic = Agents.IdealValueAgent(env_E1, place_cells_E1)
    optimal_value_function = ideal_critic.getValueFunction()

    scaling_factor = 1.0/(1 - critic_E1.getDiscountFactor())
    # Graphics.showImage(optimal_value_function, xticks=range(1,nx), yticks=range(1,ny), range=(maze.NON_GOAL_STATE_REWARD, scaling_factor * maze.GOAL_STATE_REWARD))
    Graphics.showImage(optimal_value_function, xticks=range(1,nx), yticks=range(1,ny), \
        range=(env_E1.NON_GOAL_STATE_REWARD, scaling_factor * env_E1.GOAL_STATE_REWARD))

    input('Press return for Value Function of E2')
    canvas_E2.plotValueFunction(place_cells_E2, critic, continuous=True)
    canvas_E2.plotValueFunction(place_cells_E2, critic_E2, continuous=True)
    canvas_E2.plotValueFunction(place_cells_E2, critic_E1, continuous=True)

    # Plot the ideal value function
    ideal_critic = Agents.IdealValueAgent(env_E2, place_cells_E2)
    optimal_value_function = ideal_critic.getValueFunction()

    scaling_factor = 1.0/(1 - critic_E2.getDiscountFactor())
    # Graphics.showImage(optimal_value_function, xticks=range(1,nx), yticks=range(1,ny), range=(maze.NON_GOAL_STATE_REWARD, scaling_factor * maze.GOAL_STATE_REWARD))
    Graphics.showImage(optimal_value_function, xticks=range(1,nx), yticks=range(1,ny), \
        range=(env_E2.NON_GOAL_STATE_REWARD, scaling_factor * env_E2.GOAL_STATE_REWARD))
    input('Press any key to exit!')
    
    # joint_components = Graphics.showDecomposition(weights)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')
