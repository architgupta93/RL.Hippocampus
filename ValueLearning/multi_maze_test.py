import Hippocampus
import Environment
import ValueLearning

def testMaze():
    """
    No comments here. Look at single_maze_test.py for more details!
    """
    ValueLearning.DBG_LVL = 1

    nx = 10
    ny = 10

    n_fields = round(1.0 * nx * ny)
    n_cells  = Hippocampus.N_CELLS_PER_FIELD * n_fields

    n_training_trials = 20
    n_navigation_trials = 4

    n_alternations = 4
    max_nav_steps = 100
    max_train_steps = 1000

    # Same place fields for both the environments
    environment  = Environment.RandomGoalOpenField(nx, ny)
    place_fields = Hippocampus.setupPlaceFields(environment, n_fields)

    # First Environment: Has its own place cells
    place_cells_E1 = Hippocampus.assignPlaceCells(n_cells, place_fields)

    # Second Environment: This has a different set (but the same number) of
    # place fields
    place_cells_E2 = Hippocampus.assignPlaceCells(n_cells, place_fields)

    # First, train an agent on the first environment
    (actor, critic) = ValueLearning.learnValueFunction(n_training_trials, environment, place_cells_E1, max_steps=max_train_steps)

    for alt in range(n_alternations):
        print('Alternation: %d' % alt)
        # First look at the performance of the agent in the task before it is
        # allowed to learn anything. Then allow learning
        ValueLearning.navigate(n_navigation_trials, environment, place_cells_E2, actor, critic, max_nav_steps)
        (actor, critic) = ValueLearning.learnValueFunction(n_training_trials, environment, place_cells_E2, actor, critic, max_train_steps)

        # Repeat for environment 1
        ValueLearning.navigate(n_navigation_trials, environment, place_cells_E1, actor, critic, max_nav_steps)
        (actor, critic) = ValueLearning.learnValueFunction(n_training_trials, environment, place_cells_E1, actor, critic, max_train_steps)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')