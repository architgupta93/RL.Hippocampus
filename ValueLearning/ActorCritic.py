import numpy as np
import random

class Actor(object):
    def __init__(self, actions, pfs):
        """
        Actor, takes in a place field activities and produces an action based
        on them
        """
        self._actions = actions
        self._n_actions = len(actions)
        self._weights = np.zeros((len(actions), len(pfs)), dtype=float)

        pass
    
    def getAction(self, activity):
        action_weights = np.exp(np.dot(self._weights, activity))

        # Return the maximally weighted action
        # selected_action = np.argmax(action_weights)

        # Normalize probabilities, convert to CDF and draw a uniform random
        # variable to sample from among these actions
        normalized_prob = action_weights / sum(action_weights)

        # Because of the way lists work in python, selection_cdf is the same
        # list as normalized_prob (its not a copy). Don't worry, just don't
        # reuse it later
        selection_cdf   = normalized_prob
        for ac in range(1, self._n_actions):
            selection_cdf[ac] = selection_cdf[ac] + selection_cdf[ac-1]

        uniform_sample  = np.random.rand()
        selected_action = np.searchsorted(selection_cdf, uniform_sample)

        return self._actions[selected_action]

class Critic(object):
    def __init__(self, pfs):
        """
        Critic, takes in place field activities and produces an estimate for the
        current 'value' based on these
        """
        self._weights = np.zeros(len(pfs), dtype=float)
        self._is_learning = False
    
    def getValue(self, activity):
        return np.dot(self._weights, activity)

    def updateValue(self, activity, reward):
        """
        If for a give activity, the agent received a certain reward, how
        should the value function be updated
        """

        if self._is_learning:
            raise NotImplementedError
        else:
            # Learnt values have been frozen
            return