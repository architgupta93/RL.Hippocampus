import numpy as np
import random

class Agent(object):
    """
    A general concept  of an agent that interacts with cell activity.
    The output of the agent is a weighted sum of the input activity
    """

    INITIAL_WEIGHT_VAR = 0.00

    def __init__(self, n_fields):
        # Learning parameters
        self._weight_scaling = 0.005
        self._weights   = np.zeros(())
        self._n_fields  = n_fields
        self._is_learning = True

    def getWeights(self):
        return self._weights

    def getValue(self, activity):
        return np.dot(self._weights, activity)

    def getNFields(self):
        return self._n_fields

    def setLearning(self):
        self._is_learning = True

    def unsetLearning(self):
        self._is_learning = False

    def isLearning(self):
        return self._is_learning

class Actor(Agent):
    EPSILON = 1e-8

    def __init__(self, actions, n_fields):
        """
        Actor, takes in a place field activities and produces an action based
        on them
        """
        super(Actor, self).__init__(n_fields)
        self._actions = actions
        self._n_actions = len(actions)
        self._last_selected_action = None
        self._weight_scaling = 0.01
        self._weights   = self.INITIAL_WEIGHT_VAR * np.random.randn(self._n_actions, self._n_fields)

        # UPDATE: Instead of relying only on the current activity input, we are
        # now assuming some hysterisis. This means that at the current time
        # point, a decision is made by accumulating the present and the
        # previous activities. Previous activity is scaled down by a scalar
        # factor which is also a part of the class now.
        self._previous_activity = None
        self._memory_factor = 0.5

        # UPDATE: The action that was previously chosen (say E) gets a bump in
        # its probability mimicking a 'momentum' term. It just captures the
        # fact that animals probably like to keep going in one direction.
        self._momentum_factor = 1.01
    
    def getAction(self, activity):
        # Experimenting with other monotonic functions
        baseline_activity = self.getValue(activity) 
        scaled_activity = (baseline_activity + self.EPSILON) / max(abs(baseline_activity) + self.EPSILON)

        # Method: 01
        # Include the memory effect
        if self._previous_activity is not None:
            scaled_activity += self._memory_factor * self._previous_activity
        self._previous_activity = scaled_activity

        # Method: 02
        # Go by the behavior and give more weight to the last selected action
        if self._last_selected_action is not None:
            scaled_activity[self._last_selected_action] *= self._momentum_factor

        action_weights  = np.exp(scaled_activity)

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
        if not self._is_learning:
            return

        last_action = self._last_selected_action
        for pf in range(self._n_fields):
            self._weights[last_action, pf] += self._weight_scaling * prediction_error * activity[pf]

class RandomAgent(Actor):
    """
    A random agent for testing Value updates independent of the Actor's
    policies for exploring the state-space
    """

    def __init__(self, actions, n_fields):
        super(RandomAgent, self).__init__(actions, n_fields)
        self._is_learning = False

    def getAction(self, pf_activity):
        return random.sample(self._actions, 1)[0]


class Critic(Agent):
    def __init__(self, n_fields):
        """
        Critic, takes in place field activities and produces an estimate for the
        current 'value' based on these
        """
        super(Critic, self).__init__(n_fields)

        # Learning parameters, including the proportionality constant with
        # which weights are scaled for the critic
        self.INITIAL_WEIGHT_VAR = 0.0001
        self._learning_rate   = 0.02
        self._discount_factor = 1.00

        # Weights to map place field activities to value
        self._weights  = self.INITIAL_WEIGHT_VAR * np.random.randn(self._n_fields)

    def updateValue(self, activity, new_activity, reward):
        """
        After taking an action, agent receives a reward and activity changes.
        Based on the new activity, known value of the past state, and
        received reward, we can update the value of the past state.
        """
        if not self._is_learning:
            return

        next_value = self.getValue(new_activity)
        past_value = self.getValue(activity)
        
        prediction_error = reward + self._discount_factor * next_value - \
            past_value
        
        # Set the current value to be the past value

        if self._is_learning:
            for pf in range(self._n_fields):
                self._weights[pf] += self._weight_scaling * prediction_error * activity[pf]
        
        return prediction_error

class IdealValueFunction(Critic):
    """
    Instead of computing the Value function iteratively, this generates the
    ideal value function that a Critic should converge to. Good for checking
    correctness.
    """

    def __init__(self, n_fields):
        super(IdealValueFunction, self).__init__(n_fields)