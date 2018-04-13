import Hippocampus
import Environment
import ValueLearning
import Graphics

def testMaze():
    ValueLearning.DBG_LVL = 1

    # Create a Maze for the experiment
    nx = 10
    ny = 10

    # Every location has an associated place field
    # TODO: Play around with having more/fewer place fields!
    n_fields = round(1.0 * (nx * ny)) 
    n_cells  = Hippocampus.N_CELLS_PER_FIELD * n_fields

    # Number of trials used in the experiments
    nT = 20 # Training trials
    nN = 20  # Navigation trials

    # Build the maze
    maze  = Environment.RandomGoalOpenField(nx, ny)
    canvas = Graphics.MazeCanvas(maze)

    # Generate a set of place fields for the environment
    place_fields = Hippocampus.setupPlaceFields(maze, n_fields) 
    place_cells  = Hippocampus.assignPlaceCells(n_cells, place_fields)

    # Each place field will be assigned to multiple place cells (selected randomly)
    canvas.visualizePlaceFields(place_cells)

    # Learn how to navigate this Environment
    (actor, critic) = ValueLearning.learnValueFunction(nT, maze, place_cells)

    # Try a single trial on the same Maze and see how we do
    ValueLearning.DBG_LVL = 0
    ValueLearning.navigate(nN, maze, place_cells, actor, critic, max_steps=200)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')