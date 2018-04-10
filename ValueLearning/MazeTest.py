from Environment import Maze
from ActorCritic import Actor, Critic

def __main__():
    # Create a Maze for the experiment
    nx = 100
    ny = 100

    maze  = Maze(nx, ny)
    actor = Actor(maze.getActions()) 