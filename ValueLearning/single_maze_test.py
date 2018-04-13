import Hippocampus
import Environment
import ValueLearning

def testMaze():
    # Create a Maze for the experiment
    nx = 10
    ny = 10

    # Every location has an associated place field
    # TODO: Play around with having more/fewer place fields!
    nf = nx * ny
    nT = 20

    # Build the maze
    maze  = Environment.RandomGoalOpenField(nx, ny)

    # Generate a set of place fields for the environment
    place_fields = Hippocampus.setupPlaceFields(maze, nf) 

    # Learn how to navigate this Environment
    (actor, critic) = ValueLearning.learnValueFunction(nT, maze, place_fields)

    # Stop actor and critic from learning
    actor.unsetLearning()
    critic.unsetLearning()

    # Try a single trial on the same Maze and see how we do
    (actor, critic) = ValueLearning.learnValueFunction(nT, maze, place_fields, actor, critic)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')