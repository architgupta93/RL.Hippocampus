import numpy as np
import random

class Maze(object):
    """
    Defines a rectangular maze for a navigation task
    """

    # There can be a reward(s) associated with both the goal and non-goal states.
    GOAL_STATE_REWARD = 1
    NON_GOAL_STATE_REWARD = 0

    def __init__(self, nx, ny):
        self._nx = nx
        self._ny = ny
        self._state = [0, 0]
        self._action_map = {'E':(-1, 0), 'W':(1,0), 'N':(0,1), 'S':(0,-1)}
        self._n_states = self._nx * self._ny
    
        # Placeholders for the goal location(s) and initial location(s)
        self._goal_locations = []
        self._init_locations = []

    def getBounds(self):
        return(0, 0, self._nx, self._ny)

    def getStates(self):
        """
        Return all possible states
        """

        all_states = []
        for px in range(self._nx):
            for py in range(self._ny):
                all_states.append((px, py))
        
        return all_states

    def getActions(self):
        return list(self._action_map.keys())

    def getLegalActions(self, state):
        raise NotImplementedError

    def convertActionToTranslation(self, action):
        if action in self._action_map:
            translation = self._action_map[action]
        
        no_movement = (0, 0)
        # Making use of the fact that we can only move along 1 direction. This
        # means that if we are right next to one of the boundaries, no movement
        # with happen! Correcting for these boundary cases:

        # Left X boundary
        if self._state[0] == 0:
            if translation[0] < 0:
                return no_movement

        # Right X boundary
        if self._state[0] == self._nx:
            if translation[0] > 0:
                return no_movement
        
        # Bottom Y boundary
        if self._state[1] == 0:
            if translation[1] < 0:
                return no_movement

        # Top Y boundary
        if self._state[1] == self._ny:
            if translation[1] > 0:
                return no_movement

        # TODO: Can we run into a never ending loop because of this?
        return translation

    def move(self, action):
        """
        Takes in an 'action' and performs it on the current state to generate
        a new state. Returns the reward obtained by taking this action.

        INPUTS:
            :action: An operation to be performed on the current state
        """

        translation = self.convertActionToTranslation(action)
        self._state[0] += translation[0]
        self._state[1] += translation[1]

        return self.getReward()
    
    def getCurrentState(self):
        return list(self._state)

    def getNStates(self):
        return self._n_states
    
    def getReward(self):
        if self.reachedGoalState():
            return self.GOAL_STATE_REWARD
        return self.NON_GOAL_STATE_REWARD

    def draw(self):
        """
        Draw the current state of the maze
        """

        # Select the  appropriate figure window
        if self._fig_num < 0:
            fig = plt.figure()
            self._fig_num = fig.number
        else:
            plt.figure(self._fig_num)

        # Draw the current location of the agent
        plt.scatter(self._state[0], self._state[1], marker='s', alpha=0.5, c='blue')

        # Draw the current goal location(s)
        gx, gy = zip(*self._goal_locations)
        plt.scatter(gx, gy, c='green', marker='o', alpha=0.5)

        plt.show()

    # Abstract functions to be implemented by child classes
    def redrawInitLocation(self):
        raise NotImplementedError

    def redrawGoalLocation(self):
        raise NotImplementedError

    def reachedGoalState(self):
        raise NotImplementedError

class RandomGoalOpenField(Maze):
    def __init__(self, nx, ny):
        # Call the parent class constructor
        super(RandomGoalOpenField, self).__init__(nx, ny)
        self.redrawGoalLocation()
        self.redrawInitLocation()
        return

    def redrawGoalLocation(self):
        # Have a single, random goal location inside the maze
        self._goal_locations = [(np.random.randint(0, self._nx), np.random.randint(0,self._ny))]

        # Print the goal location
        goal_location = self._goal_locations[0]
        print('Goal location: (%d, %d)' % (goal_location[0], goal_location[1]))

        return

    def redrawInitLocation(self):
        # Single, random start location inside the maze
        self._init_locations = [(np.random.randint(0, self._nx), np.random.randint(0, self._ny))]

        # Pick on the the initial locations
        initial_state = random.sample(self._init_locations, 1)[0]
        self._state[0] = initial_state[0]
        self._state[1] = initial_state[1]

        return
    
    def reachedGoalState(self):
        # There is just ONE goal location, nothing complicated here
        goal_location = self._goal_locations[0]
        return ((self._state[0] == goal_location[0]) and (self._state[1] == goal_location[1]))