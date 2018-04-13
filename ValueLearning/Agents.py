import numpy as np
import random

class Agent(object):
    """
    A general concept  of an agent that interacts with cell activity.
    The output of the agent is a weighted sum of the input activity
    """

    def __init__(self, fields):
        # Learning parameters
        self._weight_scaling = 0.005
        self._weights   = np.zeros(())
        self._n_fields  = len(fields)

    def getValue(self, activity):
        return np.dot(self._weights, activity)

    def getNFields(self):
        return self._n_fields

class Actor(Agent):
    def __init__(self, actions, pfs):
        """
        Actor, takes in a place field activities and produces an action based
        on them
        """
        super(Actor, self).__init__(pfs)
        self._actions = actions
        self._n_actions = len(actions)
        self._last_selected_action = -1
        self._weights   = np.zeros((self._n_actions, self._n_fields), dtype=float)
    
    def getAction(self, activity):
        action_weights = np.exp(self.getValue(activity))

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
        self._last_selected_action = selected_action

        return self._actions[selected_action]
    
    def updateWeights(self, activity, prediction_error):
        """
        Update the weights that are used for making action decisions
        """

        last_action = self._last_selected_action
        for pf in range(self._n_fields):
            self._weights[last_action, pf] += self._weight_scaling * prediction_error * activity[pf]

class Critic(Agent):
    def __init__(self, pfs):
        """
        Critic, takes in place field activities and produces an estimate for the
        current 'value' based on these
        """
        super(Critic, self).__init__(pfs)

        # Learning parameters, including the proportionality constant with
        # which weights are scaled for the critic
        self._learning_rate   = 0.02
        self._discount_factor = 0.9

        # Weights to map place field activities to value
        self._weights  = np.zeros(self._n_fields, dtype=float)
        self._is_learning = True

    def updateValue(self, activity, new_activity, reward):
        """
        After taking an action, agent receives a reward and activity changes.
        Based on the new activity, known value of the past state, and
        received reward, we can update the value of the past state.
        """
        next_value = self.getValue(new_activity)
        past_value = self.getValue(activity)
        
        prediction_error = reward + self._discount_factor * next_value - \
            past_value
        
        # Set the current value to be the past value

        if self._is_learning:
            for pf in range(self._n_fields):
                self._weights[pf] += self._weight_scaling * prediction_error * new_activity[pf]
        
        return prediction_error