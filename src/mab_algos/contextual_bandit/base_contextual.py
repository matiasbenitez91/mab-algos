import numpy as np

class ContextualBandit:
    """
    Base Class for Contextual Multi-Armed Bandit

    """
    def __init__(self, n_arms, context_dim, chunks="auto"):
        self.n_arms=n_arms
        self.chunks=chunks
        self.context_dim=context_dim
        self.cum_rewards=np.zeros(self.n_arms)
        self.n_iter=0
        self.arm_rounds=np.zeros(self.n_arms)

    def action(self, context):
        pass
    def update(self, action, context, reward):
        pass
