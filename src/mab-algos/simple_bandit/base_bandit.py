import dask.array as da
import numpy as np

class SimpleBandit:
    """
    Base Class for Non-contextual Multi-armed Bandit

    Parameters
    ----------
    n_arms : int
        number of arms
    chunks : int
        chunks size for dask array


    Attributes
    ----------
    cum_rewards : np array, size (n_arms)
        array with cummulative rewards of
    n_iter : int
        Description of attribute `n_iter`.
    n_arms: int
        Number of arms/bandits
    chunks: int
        chunks size for dask array
    arm_rounds: dask aarray, size=(n_arms,)
        Number of every arm was played


    """
    def __init__(self, n_arms, chunks="auto"):
        self.n_arms=n_arms
        self.chunks=chunks
        self.cum_rewards=np.zeros(self.n_arms)
        self.n_iter=0
        self.arm_rounds=np.zeros(self.n_arms)

    def action(self):
        pass

    def update(self, action, reward):
        self.cum_rewards[action]+=reward
        self.arm_rounds[action]+=1
        self.n_iter+=1
