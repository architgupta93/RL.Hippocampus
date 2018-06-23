import numpy as np
import random
import Graphics

class Agent(object):
    """
    A general concept  of an agent that interacts with cell activity.
    The output of the agent is a weighted sum of the input activity
    """

    INITIAL_WEIGHT_VAR = 0.00001

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
        self._memory_factor = 0.0

        # UPDATE: The action that was previously chosen (say E) gets a bump in
        # its probability mimicking a 'momentum' term. It just captures the
        # fact that animals probably like to keep going in one direction.
        self._momentum_factor = 2.0
    
    def getAction(self, activity):
        # Experimenting with other monotonic functions
        baseline_activity = self.getValue(activity) 
        scaled_activity = (baseline_activity + self.EPSILON) / max(abs(baseline_activity) + self.EPSILON)

        # Method: 01
        # Include the memory effect
        """
        if self._previous_activity is not None:
            scaled_activity += self._memory_factor * self._previous_activity
        self._previous_activity = scaled_activity
        """

        action_weights  = np.exp(scaled_activity)

        # Method: 02 - This should happen after the numbers have been converted
        # to probabilities (otherwise negative numbers become even more
        # negative and therefore less likely)
        # Go by the behavior and give more weight to the last selected action
        if self._last_selected_action is not None:
            action_weights[self._last_selected_action] *= self._momentum_factor

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
        self.INITIAL_WEIGHT_VAR = 0.00001
        self._learning_rate   = 200.0/self._n_fields
        self._discount_factor = 0.99

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
                self._weights[pf] += self._learning_rate * prediction_error * activity[pf]
        
        return prediction_error

class IdealValueAgent(Critic):
    """
    Instead of computing the Value function iteratively, this generates the
    ideal value function that a Critic should converge to. Good for checking
    correctness.
    """

    def __init__(self, environment, place_cells):
        super(IdealValueAgent, self).__init__(len(place_cells))

        # Get the transition matrix for the environment
        self._dims  = environment.getNStates()
        self._t_mat = environment.getTransitionMatrix()
        self._r_vec = environment.getRewardVector()

        # Debug: Plot the transition matrix and the reward vector
        Graphics.showImage(self._t_mat)
        Graphics.showImage(np.reshape(self._r_vec, self._dims))

    def getValue(self):
        raise NotImplementedError()

    def getValueFunction(self):
        """
        Calculate the value function for all the states and return it as a matrix.
                        v_fun = T * (r + d * v_fun),
                        (I - dT) * v_fun = T*r

                        v_fun: Value Function
                        T: Transition matrix
                        r: Reward vector
                        d: discount factor
        """
        b_val = np.matmul(self._t_mat, self._r_vec) # In Ax = b
        v_fun = np.linalg.solve(np.eye(len(self._r_vec)) - self._discount_factor * self._t_mat, b_val)
        return np.reshape(v_fun, self._dims)

class IdealActor(Agent):
    """
    Instead of maintaining its own set of weights, this actor accesses the
    weights of the critic and chooses the optimal action based on the current
    value function.
    """
    def __init__(self, environment, critic, place_cells):
        super(IdealActor, self).__init__(len(place_cells))
        self._actions = environment.getActions()
        self._n_actions = len(self._actions)
        # This has a copy of the environment and place cells, which it can use
        # to simulate future actions and use that to make the optimal choices
        # (based on the stored critic's value function)

        # NOTE: Keep in mind that such an actor is quite unrealistic and is
        # only being used to test the critic independent of the actor.

        self._c_environment = environment
        self._c_critic = critic
        self._c_place_cells = place_cells

    def getAction(self, activity):
        # Ignore the activity, use the environment to get the current location
        current_location = self._c_environment.getCurrentState()
        target_values    = np.zeros(self._n_actions,)
        for idx, action in enumerate(self._actions):
            translation = self._c_environment.convertActionToTranslation(action)
            next_location = (current_location[0] + translation[0], current_location[1] + translation[1])
            target_values[idx] = self._c_critic.getValue([pc.getActivity(next_location) for pc in self._c_place_cells])

        optimal_action = np.argmax(target_values)
        return self._actions[optimal_action]

    def updateWeights(self, activity, prediction_error):
        # NOTE: Function is here only for completeness. Does nothing!
        return