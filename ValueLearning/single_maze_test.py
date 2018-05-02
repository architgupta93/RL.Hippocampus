import Hippocampus
import Environment
import ValueLearning
import Graphics

def testMaze():
    ValueLearning.DBG_LVL = 2
    # Create a Maze for the experiment
    nx = 20
    ny = 20

    # Every location has an associated place field
    # TODO: Play around with having more/fewer place fields!
    nf = round(0.5 * (nx * ny))
    nT = 20 # Training trials
    nN = 20  # Navigation trials

    # Build the maze
    maze  = Environment.RandomGoalOpenField(nx, ny)
    canvas = Graphics.MazeCanvas(maze)

    # Generate a set of place fields for the environment
    place_fields = Hippocampus.setupPlaceFields(maze, nf) 
    canvas.visualizePlaceFields(place_fields)

    # Learn how to navigate this Environment
    (actor, critic) = ValueLearning.learnValueFunction(nT, maze, place_fields)

    # Try a single trial on the same Maze and see how we do
    ValueLearning.DBG_LVL = 0
    ValueLearning.navigate(nN, maze, place_fields, actor, critic, max_steps=200)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')