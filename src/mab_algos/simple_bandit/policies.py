import numpy as np
import dask.array as da


def greedy_policy(cum_rewards, arm_rounds):
    """
    Parameters
    ----------
    n_rewards : dask array
    arm_round : dask array
    Returns
    -------
    int
        index of chosen arm

    """
    max_reward=(cum_rewards/arm_rounds).max()
    return np.random.choice(np.argwhere((cum_rewards/arm_rounds)==max_reward).reshape(-1,))


def random_policy(n_arms):
    """
    Parameters
    ----------
    n_arms : int

    Returns
    -------
    int
        index of chosen arm

    """
    return np.random.randint(low=0, high=n_arms)

def egreedy_policy(epsilon, cum_rewards, arm_rounds):
    """

    Parameters
    ----------
    epsilon : float
    n_rewards : dask array
    arm_round : dask array
    Returns
    -------
    int
        index of chosen arm
    """
    if np.random.rand() <= epsilon:
        return random_policy(len(cum_rewards))
    else:
        return greedy_policy(cum_rewards, arm_rounds)

def ucb1_policy(cum_rewards, arm_rounds, n_iter):

    """

    Parameters
    ----------
    epsilon : float
    n_rewards : dask array
    arm_round : dask array
    Returns
    -------
    int
        index of chosen arm
    """
    bounds= cum_rewards/arm_rounds + np.sqrt( 2 * np.log(n_iter) / arm_rounds)
    max_bounds=bounds.max()

    return np.random.choice(np.argwhere(bounds==max_bounds).reshape(-1,))



def softmax_policy(cum_rewards, arm_rounds):
    """
    Parameters
    ----------
    n_rewards : dask array
    arm_round : dask array
    Returns
    -------
    int
        index of chosen arm

    """
    mean_reward=cum_rewards/arm_rounds
    p=np.exp(mean_reward)/np.sum(np.exp(mean_reward))
    return np.random.choice(np.arange(len(arm_rounds)), p=p)
