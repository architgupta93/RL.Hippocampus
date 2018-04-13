import Hippocampus
import Agents
import Graphics
import numpy as np

DBG_LVL = 0

def learnValueFunction(n_trials, environment, place_fields):
    """
    Main function responsible for learning value function for a given environment
    INPUTS:
    -------
    n_trials: (INTEGER) Number of trials allowed on the task
    environment: (Maze) Physical space in which the task has to be learnt
    place_fields: (PlaceField) Entity that encodes a particular location

    OUTPUTS:
    --------
    actor: (Actor Class) Entity that learns actions for a given state
    critic: (Critic Class) Entity that evaluates the value for a
        particular state. These values are used for taking actions.
    """

    # Set up the actor and critic based on the place fields
    actor = Agents.Actor(environment.getActions(), place_fields) 
    critic = Agents.Critic(place_fields)

    # Path is visualized using a graphics object
    canvas = Graphics.MazeCanvas(environment)

    n_steps  = np.zeros(n_trials, dtype=int)
    for trial in range(n_trials):
        environment.redrawInitLocation()
        while not environment.reachedGoalState():
            n_steps[trial] += 1
            current_state = environment.getCurrentState()
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
            canvas.plotValueFunction(place_fields, critic)
            print('Ended trial %d in %d steps.' % (trial, n_steps[trial]))
        
    Graphics.plot(n_steps)
    return(actor, critic)