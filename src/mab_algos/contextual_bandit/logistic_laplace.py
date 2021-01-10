from .base_contextual import ContextualBandit
import numpy as np
from mab_algorithms.simple_bandit.policies import random_policy
from scipy.optimize import minimize
from collections.abc import Iterable
from mab_algorithms.experiment.contextual.helpers.contextual_dataset import ContextualDataset


class OnlineLogisticRegression:
    """Logistic Regression with Online batch update based on https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf

    :param type lambda_: regularization parameter.
    :param type alpha: control uncertainty estimates larger values implies more exploration.
    :param type n_dim: dimension of covariates
    :attr type m: estimate of the mean of the weights
    :attr type q: estimate of variance of the weights
    :attr type w: weights
    :attr lambda_:
    :attr alpha:
    :attr n_dim:

    """

    # initializing
    def __init__(self, lambda_, alpha, n_dim):
        # the only hyperparameter is the deviation on the prior (L2 regularizer)
        self.lambda_= lambda_; self.alpha = alpha
        # initializing parameters of the model
        self.n_dim = n_dim,
        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_

        # initializing weights
        self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)

    # the loss function
    def loss(self, w, *args):
        X, y = args
        #return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum(np.log(1 + np.exp(-y * X.dot(w))))

    # the gradient
    def grad(self, w, *args):
        X, y = args
        #return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)
        return self.q * (w - self.m) + (-1) * np.multiply(np.multiply(X, y.reshape(-1,1)), 1/(1. + np.exp(y * X.dot(w))).reshape(-1,1)).sum(axis=0)

    # method for sampling weights
    def get_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)

    # fitting method
    def fit(self, X, y):

        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':True}).x
        self.m = self.w

        # step 2, update q
        P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)

    # probability output method, using weights sample
    def predict_proba(self, X, mode='sample'):
        # adding intercept to X
        #X = add_constant(X)
        # sampling weights after update
        self.w = self.get_weights()
        # using weight depending on mode
        if mode == 'sample':
            w = self.w # weights are samples of posteriors
        elif mode == 'expected':
            w = self.m # weights are expected values of posteriors
        else:
            raise Exception('mode not recognized!')
        # calculating probabilities
        proba = 1 / (1 + np.exp(-1 * X.dot(w)))
        return np.array([1-proba , proba]).T

class LogisticLaplace_Bandit(ContextualBandit):
    """Logistic Regression Thompson_Sampling

    :param type n_arms: int, number fo arms
    :param type context_dim: int, dimesion of weight space
    :param type lambda: float (0,1), regularization parameter
    :param type alpha: float (0,1), control exploration
    :param type chunks:
    :attr lambda:
    :attr alpha:
    :attr n_arms:
    :attr context_dim:

    """
    def __init__(self, n_arms, context_dim, lambda_=1.0, alpha=1, chunks="auto"):
        super().__init__(n_arms, context_dim, chunks)
        self.lambda_=lambda_
        self.alpha=alpha
        self.n_arms=n_arms
        self.context_dim=context_dim
        self.models=[OnlineLogisticRegression(self.lambda_, self.alpha, self.context_dim) for x in range(self.n_arms)]
        self.data_h = ContextualDataset(self.context_dim,self.n_arms,intercept=False)

    def update(self, action, context, reward):
        if not isinstance(reward, Iterable):
            reward=np.array([reward])
        if len(context.shape)<2:
            context=context.reshape(1,-1)
        self.data_h.add(context, action, reward)
        x, y = self.data_h.get_data(action)
        self.models[action].fit(x, y)
        self.cum_rewards[action]+=reward
        self.arm_rounds[action]+=1



class LogisticLaplace_TS(LogisticLaplace_Bandit):
    def __init__(self, n_arms, context_dim, lambda_=1.0, alpha=1, chunks="auto"):
        super().__init__(n_arms, context_dim, lambda_, alpha, chunks)
        self.name="LogisticLaplace_TS"
    def action(self, context):
        pred_proba=[x.predict_proba(context)[1] for x in self.models]
        return np.argmax(pred_proba)


class LogisticLaplace_EGreedy(LogisticLaplace_Bandit):
    def __init__(self, n_arms, context_dim,epsilon, lambda_=1.0, alpha=1, chunks="auto"):
        super().__init__(n_arms, context_dim, lambda_, alpha, chunks)
        self.epsilon=epsilon
        self.name="LogisticLaplace_EGreedy_{}".format(epsilon)
    def action(self, context):
        pred_proba=[x.predict_proba(context, mode="expected")[1] for x in self.models]
        if np.random.rand() <= self.epsilon:
            return random_policy(self.n_arms)
        else:
            return np.argmax(pred_proba)
