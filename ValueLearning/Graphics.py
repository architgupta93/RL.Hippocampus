from MotionAnimation.PY import data_types as GR
from Environment import Maze

class MazeCanvas(object):
    """
    Used for visualizing the current view of the state space
    """

    def __init__(self, maze):
        self._anim_obj = GR.Trajectory__2D()

        # Fake timestamp needed by the trajectory classes
        self._t_stamp  = 0

        maze_bounds    = maze.getBounds()
        self._min_x    = maze_bounds[0]
        self._max_x    = maze_bounds[1]
        self._min_y    = maze_bounds[2]
        self._max_y    = maze_bounds[3]
    
    def update(self, next_state):
        self._anim_obj.update(self._t_stamp, next_state[0], next_state[1])
        self._t_stamp += 1