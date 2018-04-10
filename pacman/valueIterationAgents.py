# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        iter_idx = 0
        while (iter_idx < self.iterations):
            # DEBUG
            print 'Iteration: ', iter_idx
            # Create a copy of the values for doing a batch update
            new_values = util.Counter()

            # Update the value for each of the states in the MDP
            for state in self.mdp.getStates():
                # Get the optimal action directly - We are running some code
                # redundantly here in order to avoid code duplication
                optimal_action = self.computeActionFromValues(state)
                if optimal_action is None:
                    # No more actions are possible from here, so value iteration has settled on this on
                    new_values[state] = self.values[state]
                else:
                    new_values[state] = self.computeQValueFromValues(state, optimal_action)

            self.values = new_values.copy()
            iter_idx += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        state_probs    = self.mdp.getTransitionStatesAndProbs(state, action)
        action_outcome = 0
        for possible_outcomes in state_probs:
            action_outcome += possible_outcomes[1] * \
                (self.mdp.getReward(state, action, possible_outcomes[0]) + \
                self.discount * self.values[possible_outcomes[0]])

        return action_outcome


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get possible actions from this state
        possible_actions = self.mdp.getPossibleActions(state)

        if (len(possible_actions) == 0):
            return None

        action_outcomes  = [0 for act in possible_actions]
        optimal_outcome  = -float('inf')
        optimal_idx      = None

        for idx, action in enumerate(possible_actions):
            action_outcomes[idx] = self.computeQValueFromValues(state, action)
            if (optimal_outcome < action_outcomes[idx]):
                optimal_outcome = action_outcomes[idx]
                optimal_idx     = idx

        return possible_actions[optimal_idx]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        iter_idx = 0
        iter_idx_to_states_map = self.mdp.getStates()
        n_states = len(iter_idx_to_states_map)

        # TODO: If needed, this can be updated to be more efficient.
        while iter_idx < self.iterations:
            state_to_udpate = iter_idx_to_states_map[iter_idx%n_states]
            best_action     = self.computeActionFromValues(state_to_udpate)
            if best_action is not None:
                self.values[state_to_udpate] = self.computeQValueFromValues(state_to_udpate, best_action)
            iter_idx += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        print 'Running value iteration for %d iterations.'% self.iterations

        # Make the priority queue
        value_update_priority = util.PriorityQueue()

        # For each state, build a set of predecessors
        predecessor_map = collections.OrderedDict()
        for state in self.mdp.getStates():
            # Get possible action from this state
            possible_actions = self.mdp.getPossibleActions(state)
            for action in possible_actions:
                outcomes = self.mdp.getTransitionStatesAndProbs(state, action)
                q_value  = 0
                max_q    = -float('inf')

                # Attach this state as a predecessor of all the outcomes
                for transition_outcome in outcomes:
                    # Everything starts at a 0 value intially
                    q_value += transition_outcome[1] * (\
                        self.mdp.getReward(state, action, transition_outcome[0]))
                    if transition_outcome[0] not in predecessor_map:
                        predecessor_map[transition_outcome[0]] = set()
                    predecessor_map[transition_outcome[0]].add(state)

                if (max_q < q_value):
                    max_q = q_value

            if not self.mdp.isTerminal(state):
                # print 'State: ', state, ' inserted with priority ', abs(max_q)
                value_update_priority.push(state, -abs(max_q))

        # DEBUG: Print the predecessor map
        # print predecessor_map

        iter_idx = 0
        while not value_update_priority.isEmpty():
            iter_idx        += 1
            if (iter_idx > self.iterations):
                break

            state_to_update  = value_update_priority.pop()
            # DEBUG
            # print 'Updating state:', state_to_update
            # print 'Previous value: ', self.values[state_to_update]

            best_action      = self.computeActionFromValues(state_to_update)
            new_q_value      = self.computeQValueFromValues(state_to_update, best_action)
            self.values[state_to_update] = new_q_value
            # print 'Updated to: ', self.values[state_to_update]
            # print

            # Look at all the predecessors of state_to_update
            for predecessor in predecessor_map[state_to_update]:
                # print 'Checking predecessor: ', predecessor
                # print 'Predecessor has value: ', self.values[predecessor]
                best_action      = self.computeActionFromValues(predecessor)
                new_q_value      = self.computeQValueFromValues(predecessor, best_action)
                diff             = abs(new_q_value - self.values[predecessor])

                # print 'This could incremeted by: ', -diff

                if diff > self.theta:
                    # print 'Pushing ', predecessor, ' onto priority queue'
                    value_update_priority.update(predecessor, -diff)
            # print

        # print
