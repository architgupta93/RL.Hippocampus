import Hippocampus
import Agents
import Graphics
import numpy as np

DBG_LVL = 0

def learnValueFunction(n_trials, environment, place_fields, actor=None, critic=None, max_steps=np.Inf):
    """
    Main function responsible for learning value function for a given environment
    INPUTS:
    -------
    n_trials: (INTEGER) Number of trials allowed on the task
    environment: (Maze) Physical space in which the task has to be learnt
    place_fields: (PlaceField) Entity that encodes a particular location

    <OPTIONAL INPUTS>
    actor: Pre-trained actor
    critic: Pre-trained critic

    OUTPUTS:
    --------
    actor: (Actor Class) Entity that learns actions for a given state
    critic: (Critic Class) Entity that evaluates the value for a
        particular state. These values are used for taking actions.
    """

    # Set up the actor and critic based on the place fields
    if actor is None:
        actor = Agents.Actor(environment.getActions(), place_fields) 
    else:
        assert(actor.getNFields() == len(place_fields))

    if critic is None:
        critic = Agents.Critic(place_fields)
    else:
        assert(critic.getNFields() == len(place_fields))

    n_steps  = np.zeros(n_trials, dtype=int)
    for trial in range(n_trials):
        # Path is visualized using a graphics object
        canvas = Graphics.MazeCanvas(environment)

        environment.redrawInitLocation()
        while not environment.reachedGoalState():
            if (n_steps[trial] > max_steps):
                break

            n_steps[trial] += 1
            current_state = environment.getCurrentState()
            canvas.update(current_state)
            if DBG_LVL > 1:
                print('On state: (%d, %d)' % (current_state[0], current_state[1]))

            # Get the place field activity based on the current location
            pf_activity = [pf.getActivity(current_state) for pf in place_fields]

            # Get an action based on the place field activity
            next_action = actor.getAction(pf_activity)
            if DBG_LVL > 1:
                print('Selected Action: %s' % next_action)

            # Apply this action onto the environment
            reward = environment.move(next_action)
            # canvas.update(environment.getCurrentState())

            # Use the obtained reward to update the value
            new_environment_state   = environment.getCurrentState()
            new_pf_activity  = [pf.getActivity(new_environment_state) for pf in place_fields]
            prediction_error = critic.updateValue(pf_activity, new_pf_activity, reward)
            actor.updateWeights(pf_activity, prediction_error)

        if (DBG_LVL > 0):
            print('Ended trial %d in %d steps.' % (trial, n_steps[trial]))
            canvas.plotTrajectory()
            if (DBG_LVL > 1):
                canvas.plotValueFunction(place_fields, critic)
        
    Graphics.plot(n_steps)
    return(actor, critic)

def navigate(n_trials, environment, place_fields, actor, critic, max_steps):
    """
    This function navigates through an environment with a given actor and a critic
    - Performs a subset of operations of learnValueFunction
    - There is some code duplication because of this but it can't be helped
    """

    actor_was_learning = False
    critic_was_learning = False

    if actor.isLearning():
        actor_was_learning = True
        actor.unsetLearning()

    if critic.isLearning():
        critic_was_learning = True
        critic.unsetLearning()

    learnValueFunction(n_trials, environment, place_fields, actor, critic, max_steps)

    if actor_was_learning:
        actor.setLearning()

    if critic_was_learning:
        critic.setLearning()