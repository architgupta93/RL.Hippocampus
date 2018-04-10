import numpy as np

class Maze(object):
    """
    Defines a rectangular maze for a navigation task
    """
    def __init__(self, nx, ny):
        self._nx = nx
        self._ny = ny
        self._state = (0, 0)
        self._action_list = ['E', 'W', 'N', 'S']
    
    def getActions(self):
        return self._action_list

    def getLegalActions(self, state):
        raise NotImplementedError

    def convertActionToTranslation(self, action):
        raise NotImplementedError

    def move(self, action):
        """
        Takes in an 'action' and performs it on the current state to generate a new state
        INPUTS:
            :action: An operation to be performed on the current state
        """

        translation = self.convertActionToTranslation(action)
        self._state[0] += translation[0]
        self._state[1] += translation[1]
        return
    
    def getCurrentState(self):
        return self._state