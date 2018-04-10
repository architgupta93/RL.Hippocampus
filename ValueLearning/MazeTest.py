import Hippocampus
import Environment
import Graphics
from ActorCritic import Actor, Critic
import numpy as np
import matplotlib.pyplot as plt

def testMaze():
    is_interactive = True

    # Create a Maze for the experiment
    nx = 10
    ny = 10
    nf = 20

    # Build the maze
    maze  = Environment.RandomGoalOpenField(nx, ny)

    # Generate a set of place fields for the environment
    place_fields = Hippocampus.setupPlaceFields(maze, nf) 

    # Set up the actor and critic based on the place fields
    actor = Actor(maze.getActions(), place_fields) 
    critic = Critic(place_fields)

    # Path is visualized using a graphics object
    canvas = Graphics.MazeCanvas(maze)

    while not maze.reachedGoalState():
        current_state = maze.getCurrentState()
        print('On state: (%d, %d)' % (current_state[0], current_state[1]))

        # Get the place field activity based on the current location
        pf_activity = [pf.getActivity(current_state) for pf in place_fields]

        # Get an action based on the place field activity
        next_action = actor.getAction(pf_activity)
        print('Selected Action: %s' % next_action)

        # Apply this action onto the environment
        reward = maze.move(next_action)
        canvas.update(maze.getCurrentState())

        # Use the obtained reward to update the value
        critic.updateValue(pf_activity, reward)

    # Plot the value landscape
    states = maze.getStates()
    values = np.zeros(len(states))

    for idx, m_st in enumerate(states):
        activity = [pf.getActivity(m_st) for pf in place_fields]
        values[idx] = critic.getValue(activity)
    
    plt.plot(values)

if __name__ == "__main__":
    testMaze()
    print('Execution complete. Exiting!')