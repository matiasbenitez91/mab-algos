from .base_bandit import SimpleBandit
from .policies import (greedy_policy, egreedy_policy, random_policy, ucb1_policy, softmax_policy)
import numpy as np
import dask.array as da


class E_Greedy(SimpleBandit):
    """E_Greedy algorithm. Play Greedy with probablity 1-epsilon, Play randomly with probability epsilon.

    Parameters
    ----------
    n_arms : int
        number of arms
    epsilon : float (0,1)
        probability to play randomly
    chunks : int
        chunks size for dask array


    Attributes
    ----------
    cum_rewards : np array, size (n_arms)
        array with cummulative rewards of
    n_iter : int
        Description of attribute `n_iter`.
    n_arms: int
        number of arms
    epsilon: float (0,1)
        probability to play randomly
    chunks: int
        chunks size for dask array
    arm_rounds: dask array shape(n_arms,)
        number of rounds played for each arm

    """
    def __init__(self, n_arms, epsilon,chunks="auto"):
        super().__init__(n_arms, chunks)
        self.epsilon=epsilon

    def action(self):
        zero_arms=np.argwhere(self.arm_rounds==0).reshape(-1,)
        if len(zero_arms)>0:
            return np.random.choice(zero_arms)
        else:
            return egreedy_policy(self.epsilon, self.cum_rewards, self.arm_rounds)
    """
    @property
    def epsilon(self):
        return self.epsilon

    @epsilon.setter
    def epsilon(self, value):
        if value >1 or value<0:
            raise ValueError("'epsilon' should be in the range [0,1]")
        self.epsilon=value
    """
class UCB1(SimpleBandit):
    """Upper Confidence Bound model.

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
    epsilon: float (0,1)
    chunks: int
        chunks size for dask array
    arm_rounds: dask array shape(n_arms,)
    """
    def __init__(self, n_arms, chunks="auto"):
        super().__init__(n_arms, chunks)

    def action(self):
        zero_arms=np.argwhere(self.arm_rounds==0).reshape(-1,)
        if len(zero_arms)>0:
            return np.random.choice(zero_arms)
        else:
            return ucb1_policy(self.cum_rewards, self.arm_rounds, self.n_iter)


class Random(SimpleBandit):
    """
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
        Number of bandits
    chunks: int
        chunks size for dask array
    arm_rounds: dask array shape(n_arms,)
    """
    def __init__(self, n_arms, chunks="auto"):
        super().__init__(n_arms, chunks)

    def action(self):
        return random_policy(self.n_arms)

class Greedy(SimpleBandit):
    """
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
        number of total rounds
    n_arms: int
        number of arms
    chunks: int
        chunks size for dask array
    arm_rounds: dask array shape(n_arms,)
        number of rounds played for each arm
    """
    def __init__(self, n_arms, chunks="auto"):
        super().__init__(n_arms, chunks)

    def action(self):
        zero_arms=np.argwhere(self.arm_rounds==0).reshape(-1,)
        if len(zero_arms)>0:
            return np.random.choice(zero_arms)
        else:
            return greedy_policy(self.cum_rewards, self.arm_rounds)

class Thompson_Sampling(SimpleBandit):
    """Thompson Sampling for bounded reward and Beta prior. The data generation is as followed:
    initialize F_i, S_i =0 for i in SetArms
    for n in N_iterations:
        sample p_i~Beta(S_i+1, F_i+1)
        choose k=argmax_i(p_i)
        observe reward r
        sample R ~ Bernoulli(r)
        if R=1 then S_k+=1, else F_k=+=1


    More detailed description in Analysis of Thompson Sampling for the multi-armed bandit problem (https://arxiv.org/pdf/1111.1797.pdf)

    Parameters
    ----------
    n_arms : int
    upper_bound : float
        upper bound of rewards
    lower_bound : float
        lower bound of rewards
    chunks : int

    Attributes
    ----------
    arm_success : Dask Array
        number of bernoulli trials success
    upper_bound : float
        upper bound of rewards
    lower_bound : float
        lower bound of rewards

    """
    def __init__(self, n_arms, upper_bound, lower_bound, chunks="auto"):
        super().__init__(n_arms, chunks)
        self.upper_bound=upper_bound
        self.lower_bound=lower_bound
        self.arm_success=np.zeros(shape=(self.n_arms,))


    def action(self):
        p_sample=np.random.beta(a=self.arm_success+1, b=(self.arm_rounds-self.arm_success+1))
        return np.random.choice(np.argwhere(p_sample==p_sample.max()).reshape(-1,))

    def update(self, action, reward):
        self.cum_rewards[action]+=reward
        self.arm_rounds[action]+=1
        self.n_iter+=1
        p=(reward-self.lower_bound)/(self.upper_bound-self.lower_bound)
        if p<0 or p>1:
            raise ValueError("p has value {}, and must be in interval [0,1]".format(p))
        if np.random.binomial(n=1,p=p)==1:
            self.arm_success[action]+=1

class Bernoulli_TS(Thompson_Sampling):
    """Thompsom Sampling for Binary rewards.

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
        number of total rounds
    n_arms: int
        number of arms
    chunks: int
        chunks size for dask array
    arm_rounds: dask array shape(n_arms,)
        number of rounds played for each arm
    """

    def __init__(self, n_arms, chunks="auto"):
        super().__init__(n_arms=n_arms, upper_bound=1, lower_bound=0,chunks=chunks)


class SoftMax(SimpleBandit):
    """Policy used is softmax

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
    epsilon: float (0,1)
    chunks: int
        chunks size for dask array
    arm_rounds: dask array shape(n_arms,)
    """
    def __init__(self, n_arms, chunks="auto"):
        super().__init__(n_arms, chunks)

    def action(self):
        zero_arms=np.argwhere(self.arm_rounds==0).reshape(-1,)
        if len(zero_arms)>0:
            return np.random.choice(zero_arms)
        else:
            return softmax_policy(self.cum_rewards, self.arm_rounds)
