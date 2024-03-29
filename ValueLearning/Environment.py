import numpy as np
import random

# Graphics
import matplotlib.pyplot as plt

DEFAULT_MOVE_DISTANCE = 1.0

class Maze(object):
    """
    Defines a rectangular maze for a navigation task
    """

    # There can be a reward(s) associated with both the goal and non-goal states.
    GOAL_STATE_REWARD = 0.05
    NON_GOAL_STATE_REWARD = -0.0005

    def __init__(self, nx, ny, move_distance=None):
        # Macro-states for Place field positioning
        self._nx = nx
        self._ny = ny
        self._n_states = (2+self._nx) * (2+self._ny)
        self.MOVE_DISTANCE = DEFAULT_MOVE_DISTANCE
        if move_distance is not None:
            self.MOVE_DISTANCE = move_distance

        # Micro-states for traversal
        self._ux = round(nx/self.MOVE_DISTANCE)
        self._uy = round(ny/self.MOVE_DISTANCE)

        # State and actions
        self._state = [0, 0]
        self._action_map = {'E':(-self.MOVE_DISTANCE, 0), 'W':(self.MOVE_DISTANCE,0), 'N':(0,self.MOVE_DISTANCE), 'S':(0,-self.MOVE_DISTANCE)}
    
        # Placeholders for the goal location(s) and initial location(s)
        self._goal_locations = []
        self._init_locations = []

        # Distance from the goal at which you are declared to have reached it
        self._goal_th  = 0.5
        self._fig_num  = -1

    def getBounds(self):
        return(0, 0, self._nx, self._ny)

    def getStates(self):
        """
        Return all possible states for setting up overlaying place fields
        """

        all_states = []
        for px in range(-1,self._nx+2):
            for py in range(-1,self._ny+2):
                all_states.append((px, py))
        
        return all_states

    def getMoveStates(self):
        """
        Return all possible states that can be moved into
        """

        all_states = []
        for px in range(0,self._ux):
            for py in range(0,self._uy):
                all_states.append((px * self.MOVE_DISTANCE, py * self.MOVE_DISTANCE))
        
        return all_states

    def getActions(self):
        return list(self._action_map.keys())

    def convertActionToTranslation(self, action):
        if action in self._action_map:
            translation = self._action_map[action]
        
        no_movement = (0, 0)
        # Making use of the fact that we can only move along 1 direction. This
        # means that if we are right next to one of the boundaries, no movement
        # with happen! Correcting for these boundary cases:

        # TODO: This code is possibly wrong (Needs to be fixed)
        # Left X boundary
        if self._state[0] + translation[0] <= 0:
            return no_movement

        # Right X boundary
        if self._state[0] + translation[0] >= self._nx:
            return no_movement
        
        # Bottom Y boundary
        if self._state[1] + translation[1] <= 0:
            return no_movement

        # Top Y boundary
        if self._state[1] + translation[1] >= self._ny:
            return no_movement

        # Check if we are in the goal state
        if self.reachedGoalState():
            return no_movement

        return translation

    def move(self, action):
        """
        Takes in an 'action' and performs it on the current state to generate
        a new state. Returns the reward obtained by taking this action.

        INPUTS:
            :action: An operation to be performed on the current state
        """

        translation = self.convertActionToTranslation(action)

        # In our case, reward is contingent on which state you are in. Being in
        # the goal state here means that you will be rewarded.
        reward = self.getReward()

        self._state[0] += translation[0]
        self._state[1] += translation[1]

        return reward
    
    def getCurrentState(self):
        return list(self._state)

    def getNStates(self):
        return (self._ux-1, self._uy-1)
    
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

    def getWalls(self):
        return []

    def redrawInitLocation(self):
        # Single, random start location inside the maze
        self._init_locations = [self._getValidLocation()]
        self.resetInitLocation()

    def resetInitLocation(self):
        # Set the state to one of the initial locations that was previously selected
        initial_state = random.sample(self._init_locations, 1)[0]
        self._state[0] = initial_state[0]
        self._state[1] = initial_state[1]

    def redrawGoalLocation(self):
        # Have a single, random goal location inside the maze
        self._goal_locations = [self._getValidLocation()]
        # self._goal_locations = [(1, 2)]

        # Print the goal location
        goal_location = self._goal_locations[0]
        print('Goal location: (%d, %d)' % (goal_location[0], goal_location[1]))

        return

    def getGoalLocation(self):
        return self._goal_locations[0]

    def setup(self):
        self.redrawGoalLocation()
        self.redrawInitLocation()

    # Abstract functions to be implemented by child classes
    def _getValidLocation(self):
        raise NotImplementedError()

    def getTransitionMatrix(self):
        """
        For the discrete state system (where you hop only between the
        specified states of the system, and not in between them), get the
        transition matrix (probability of transitioning from state A to state
        B) assuming all legal transitions from a state are equally likely.
        """

        goal_state_idxs    = []
        n_reachable_states = (self._ux - 1) * (self._uy - 1)
        transition_matrix  = np.zeros((n_reachable_states, n_reachable_states), dtype=float)
        action_selection_prob = 1.0/len(self._action_map)
        for st_x in range(1, self._ux):
            for st_y in range(1, self._uy):
                self._state         = (st_x * self.MOVE_DISTANCE, st_y * self.MOVE_DISTANCE)
                current_state_idx   = ((self._uy - 1) * (st_x-1)) + (st_y-1)
                if self.reachedGoalState():
                    goal_state_idxs.append(current_state_idx)
                for action in self._action_map:
                    analog_tr       = self.convertActionToTranslation(action)
                    translation     = (round(analog_tr[0]/ self.MOVE_DISTANCE), round(analog_tr[1]/self.MOVE_DISTANCE))
                    new_state_idx   = (self._uy - 1) * (st_x + translation[0] - 1) + (st_y + translation[1] - 1)
                    if new_state_idx >= n_reachable_states:
                        # Don't know what else to do about this - It is tricky!
                        new_state_idx = current_state_idx
                    transition_matrix[current_state_idx, new_state_idx] += action_selection_prob

        # For the goal state, none of this applies, you basically stay in the
        # goal state forever. In case of smaller move distances than the matrix
        # size, there would be multiple goal lications
        reward_vec = self.NON_GOAL_STATE_REWARD * np.ones((n_reachable_states,1), dtype=float)
        for goal in goal_state_idxs:
            transition_matrix[goal, :] = 0.0
            transition_matrix[goal, goal] = 1.0
            reward_vec[goal] = self.GOAL_STATE_REWARD

        return (reward_vec, transition_matrix)

    def reachedGoalState(self):
        raise NotImplementedError()

    def getOptimalDistanceToGoal(self):
        raise NotImplementedError()

class RandomGoalOpenField(Maze):
    def __init__(self, nx, ny, move_distance=None):
        # Call the parent class constructor
        super(RandomGoalOpenField, self).__init__(nx, ny, move_distance)
        self.setup()
        return

    def _getValidLocation(self):
        # Return any point at random from the field
        return (np.random.randint(1, self._nx), np.random.randint(1,self._ny))
    
    def reachedGoalState(self):
        # There is just ONE goal location, nothing complicated here
        goal_location = self._goal_locations[0]
        return ((pow(self._state[0] - goal_location[0], 2) + pow(self._state[1] - goal_location[1], 2) < pow(self._goal_th, 2)))

    def getOptimalDistanceToGoal(self):
        goal_location = self._goal_locations[0]
        init_location = self._init_locations[0]

        # NOTE: Subtract the goal threshold because ending within that regions
        # counts as having reached the goal.
        return float(abs(goal_location[0]-init_location[0])+abs(goal_location[1]-init_location[1])) - self._goal_th

class Wall(object):
    """
    A structure in a 2D environment that can be used to obstruct free path.
    """

    # The walls are not 1D lines, they have some thickness. This is modelled by
    # a single number which is half of the thickness of the wall.
    _WALL_THICKNESSS = 1.0

    def __init__(self, start, end):
        # Class constructor: Takes the two end points of the wall (start and
        # end). Each endpoint is a set of 2 numbers (x, y).
        self._start = start
        self._end   = end

        # For the sake of simplicity, we are allowing only vertical and
        # horizontal walls. This makes detecting collisions (illegal actions) a
        # lot easier.
        self._is_vert = (self._start[0] == self._end[0])
        if (self._start[0] != self._end[0]) and (self._start[1] != self._end[1]):
            ValueError('Wall is neither vertical not horizontal. Aborting!')

    def includesPoint(self, pt):
        # Check if the wall segment includes the specified point
        if self._is_vert:
            return (((self._start[1] <= pt[1] <= self._end[1]) or (self._end[1] <= pt[1] <= self._start[1])) \
            and (abs(self._start[0] - pt[0]) <= self._WALL_THICKNESSS))

        return (((self._start[0] <= pt[0] <= self._end[0]) or (self._end[0] <= pt[0] <= self._start[0])) \
        and (abs(self._start[1] - pt[1]) <= self._WALL_THICKNESSS))

    def crosses(self, pt1, pt2):
        # Check if this wall segment crosses the segment 'other_wall'. This can
        # be useful in determining if a step being taken by an agent is legal.
        # UPDATE (2018/06/19): Taking into account that we are dealing with WALL SEGMENTS and not infinite walls

        if self._is_vert:
            # If the movement is also vertical
            if pt1[0] == pt2[0] == self._start[0]:
                return ((self._start[1] < pt1[1] < self._end[1]) or (self._start[1] < pt2[1] < self._end[1]))

            if (pt1[0] <= self._start[0] <= pt2[0]) or (pt2[0] <= self._start[0] <= pt1[0]):
                # Check that the intersection lies within the wall segment
                return ((self._start[1] < pt1[1] < self._end[1]) or (self._start[1] > pt1[1] > self._end[1])) and \
                ((self._start[1] < pt1[1] < self._end[1]) or (self._start[1] > pt1[1] > self._end[1]))
            else:
                return False

        if pt1[1] == pt2[1] == self._start[1]:
            return ((self._start[0] < pt1[0] < self._end[0]) or (self._start[0] < pt2[0] < self._end[0]))

        if (pt1[1] <= self._start[1] <= pt2[1]) or (pt2[1] <= self._start[1] <= pt1[1]):
            return ((self._start[0] < pt1[0] < self._end[0]) or (self._start[0] > pt1[0] > self._end[0])) and \
            ((self._start[0] < pt1[0] < self._end[0]) or (self._start[0] > pt1[0] > self._end[0]))
        return False

    def getPlottingData(self):
        # Return the x and y coordinates of the end points separately (more
        # convenient for plotting)
        return (self._start[0], self._end[0]), (self._start[1], self._end[1])

class MazeWithWalls(Maze):
    """
    A Maze with obstacles in the path. The open maze task is too simple to
    see any learning happening. In the absence of anything interesting,
    debugging was becoming increasingly difficult. Having a more difficult
    task would be a better way to assess what is going on.
    """
    def __init__(self, nx, ny, walls=[], move_distance=None):
        # Class constructor
        super(MazeWithWalls, self).__init__(nx, ny, move_distance)

        # Add walls corresponding to boundaries
        l_wall = Wall((0,0), (0,ny))    # Left
        r_wall = Wall((nx,0), (nx,ny))  # Right
        b_wall = Wall((0,0), (nx,0))    # Bottom
        t_wall = Wall((0,ny), (nx,ny))  # Top
        self._walls = list([l_wall, r_wall, b_wall, t_wall])

        # Add the user supplied walls
        self._walls.extend(walls)

        self.setup()
        return

    def getWalls(self):
        # Get the list of walls (needed for plotting)
        return self._walls

    def addWall(self, wall):
        self._walls.append(wall)

    def convertActionToTranslation(self, action):
        if action in self._action_map:
            translation = self._action_map[action]
        no_movement = (0, 0)
        next_naive_state = (self._state[0] + translation[0], self._state[1] + translation[1])
        for wall in self._walls:
            if wall.crosses(self._state, next_naive_state):
                return no_movement

        # Check if we are in the goal state
        if self.reachedGoalState():
            return no_movement

        return translation

    def _getValidLocation(self):
        # Get a location that is not coincident with any of the walls in the
        # maze. Can be reused for both getting final and initial locations for
        # the task.
        location = None
        is_invalid = True

        while (is_invalid):
            # Have a single, random goal location inside the maze
            is_invalid = False
            location = (1+np.random.rand()*(self._nx-1), 1+np.random.rand()*(self._ny-1))
            for wall in self._walls:
                if wall.includesPoint(location):
                    is_invalid = True
                    break
        return location
    
    def reachedGoalState(self):
        # There is just ONE goal location, nothing complicated here
        goal_location = self._goal_locations[0]
        return ((pow(self._state[0] - goal_location[0], 2) + pow(self._state[1] - goal_location[1], 2) < pow(self._goal_th, 2)))

    def getOptimalDistanceToGoal(self):
        # TODO: Will have to implement BFS here to find the optimal number of
        # steps, not sure if it is work it :(

        # NOTE: For the time being, we will just use this (INCORRECT) implementation
        goal_location = self._goal_locations[0]
        init_location = self._init_locations[0]
        return float(abs(goal_location[0]-init_location[0])+abs(goal_location[1]-init_location[1])) - self._goal_th
