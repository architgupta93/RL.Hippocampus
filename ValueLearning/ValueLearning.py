import Hippocampus
import Agents
import Graphics
import numpy as np

DBG_LVL = 0

def learnValueFunction(n_trials, environment, place_cells, actor=None, critic=None, max_steps=np.Inf):
    """
    Main function responsible for learning value function for a given environment
    INPUTS:
    -------
    n_trials: (INTEGER) Number of trials allowed on the task
    environment: (Maze) Physical space in which the task has to be learnt
    place_cells: (PlaceCell) Entity that encodes a particular location as a population

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
        # actor = Agents.Actor(environment.getActions(), len(place_cells)) 
        actor = Agents.RandomAgent(environment.getActions(), len(place_cells)) 
    else:
        assert(actor.getNFields() == len(place_cells))

    if critic is None:
        critic = Agents.Critic(len(place_cells))
    else:
        assert(critic.getNFields() == len(place_cells))

    n_steps  = np.zeros(n_trials, dtype=int)
    for trial in range(n_trials):
        # Path is visualized using a graphics object
        canvas = Graphics.WallMazeCanvas(environment)

        environment.redrawInitLocation()
        initial_state = environment.getCurrentState()
        canvas.update(initial_state)
        while not environment.reachedGoalState():
            if (n_steps[trial] > max_steps):
                break

            n_steps[trial] += 1
            current_state = environment.getCurrentState()
            if DBG_LVL > 1:
                print('On state: (%.2f, %.2f)' % (current_state[0], current_state[1]))

            # Get the place field activity based on the current location
            pf_activity = [pf.getActivity(current_state) for pf in place_cells]

            # Get an action based on the place field activity
            next_action = actor.getAction(pf_activity)
            if DBG_LVL > 1:
                print('Selected Action: %s' % next_action)

            # Apply this action onto the environment
            reward = environment.move(next_action)
            # canvas.update(environment.getCurrentState())

            # Use the obtained reward to update the value
            new_environment_state   = environment.getCurrentState()
            canvas.update(new_environment_state)

            new_pf_activity  = [pf.getActivity(new_environment_state) for pf in place_cells]
            prediction_error = critic.updateValue(pf_activity, new_pf_activity, reward)
            actor.updateWeights(pf_activity, prediction_error)

        if (DBG_LVL > 0):
            print('Ended trial %d in %d steps.' % (trial, n_steps[trial]))
            # At debug level 1, only the first and the last trajectories, and
            # corresponding value functions are shown. At higher debug levels,
            # the entire trajectory is shown for every iteration
            if (DBG_LVL > 1) or (trial == 0) or (trial == n_trials-1):
                # Plot the trajectory taken for this trial
                canvas.plotTrajectory()

                # This takes extremely long when using a population of neurons
                canvas.plotValueFunction(place_cells, critic)
        
                # Plot a histogram of the weightS
                """
                critic_weights = np.reshape(critic.getWeights(), -1)
                Graphics.histogram(critic_weights)
                """


    if (DBG_LVL > 0):
        Graphics.plot(n_steps)
    else:
        print(np.mean(n_steps), np.std(n_steps))

    return(actor, critic, n_steps)

def navigate(n_trials, environment, place_cells, actor, critic, max_steps):
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

    (_, _, n_steps) = learnValueFunction(n_trials, environment, place_cells, actor, critic, max_steps)

    if actor_was_learning:
        actor.setLearning()

    if critic_was_learning:
        critic.setLearning()

    return n_steps