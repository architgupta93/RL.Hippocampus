import Hippocampus
import Environment
import ValueLearning

def testMaze():
    """
    No comments here. Look at single_maze_test.py for more details!
    """

    nx = 10
    ny = 10
    nf = nx * ny
    n_training_trials = 20
    n_navigation_trials = 4

    n_alternations = 4
    max_nav_steps = 100
    max_train_steps = 200

    # First Environment: Has its own place fields
    environment_1  = Environment.RandomGoalOpenField(nx, ny)
    place_fields_1 = Hippocampus.setupPlaceFields(environment_1, nf)

    # Second Environment: This has a different set (but the same number) of
    # place fields
    environment_2  = Environment.RandomGoalOpenField(nx, ny)
    place_fields_2 = Hippocampus.setupPlaceFields(environment_2, nf)

    # First, train an agent on the first environment
    (actor, critic) = ValueLearning.learnValueFunction(n_training_trials, environment_1, place_fields_1, max_steps=max_train_steps)

    for alt in range(n_alternations):
        print('Alternation: %d' % alt)
        # First look at the performance of the agent in the task before it is
        # allowed to learn anything. Then allow learning
        ValueLearning.navigate(n_navigation_trials, environment_2, place_fields_2, actor, critic, max_nav_steps)
        (actor, critic) = ValueLearning.learnValueFunction(n_training_trials, environment_2, place_fields_2, actor, critic, max_train_steps)

        # Repeat for environment 1
        ValueLearning.navigate(n_navigation_trials, environment_1, place_fields_1, actor, critic, max_nav_steps)
        (actor, critic) = ValueLearning.learnValueFunction(n_training_trials, environment_1, place_fields_1, actor, critic, max_train_steps)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')