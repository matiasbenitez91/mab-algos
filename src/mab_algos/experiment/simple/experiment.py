import numpy as np
import matplotlib.pyplot as plt


class Simple_Sampler:
    """Class that Provides samples from a given data/distribution by using method `draw`

    Parameters
    ----------
    n_arms : int
    sample_from : pd DataFrame, or np array of shape (n,n_arms)
        if None provided, then it creates the array as follows:
            -sample a_i~Uniform(0,30), b_i~Uniform(0,30) for i in range(n_arms)
            - sample reward_k,i ~Beta(a_i, b_i) for i in range  n_arms for k in 2000

    Attributes
    ----------
    data : np array
    n_arms: int

    """
    def __init__(self, n_arms, sample_from=None):
        self.n_arms=n_arms

        if sample_from is None:
            a=a=np.random.uniform(low=0, high=30, size=(self.n_arms,))
            b=np.random.uniform(low=0, high=30, size=(self.n_arms,))
            self.data=np.random.beta(a=a, b=b,size=(2000, self.n_arms) )

        else:
            if sample_from.shape[1]!=n_arms:
                raise TypeError("synthetic dimension 2 must equal n_arms")
            self.data=np.array(sample_from).reshape(-1,self.n_arms)

    def draw(self):
        index=np.random.randint(len(self.data))
        sample=self.data[index,:]
        np.delete(self.data, index, 0)
        return sample




class SimpleExperiment:
    """ Class that emulates an agent playing multi-armed bandit

    Parameters
    ----------
    algos : Dict or list
        list of algorithms of the class SimpleBandit
    sampler : Simple_Sampler

    Attributes
    ----------
    cum_rewards : np array shape (n_iter, n_arms),

    algos: dict of algotiyhms of the class SimpleBandit
    sampler: Simple Sampler

    """

    def __init__(self, algos, sampler):
        if type(algos)!=dict:
            self.algos={i:value for i,value in enumerate(algos)}
        else:
            self.algos=algos
        self.sampler = sampler
        n_iter=None
        self.cum_rewards=[[0] for x in range(len(self.algos))]

    def run(self, n_iter):
        for round in range(n_iter):
            sample=self.sampler.draw()
            for i, (_,algo) in enumerate(self.algos.items()):
                arm=algo.action()
                reward=sample[arm]
                self.cum_rewards[i].append(reward)
                algo.update(arm, reward)

        self.cum_rewards=np.vstack(self.cum_rewards).T


    def plot_cum_reward(self, steps=None):
        performance_pairs = []
        plt.figure(figsize=(12,10))
        if steps is None:
            for j, name in enumerate(self.algos.keys()):
                plt.plot(range(len(self.cum_rewards)), np.cumsum(self.cum_rewards[:, j]), label=name)
        else:
            for j, name in enumerate(self.algos.keys()):
                plt.plot(range(steps), np.cumsum(self.cum_rewards[:steps, j]), label=name)
        plt.title("Comparisson Cummulative Reward")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Cummulative Reward")
        plt.show()

    def mean_reward(self):
        pass
    def frequency_arm(self):
        pass
