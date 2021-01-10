from .base_contextual import ContextualBandit
import numpy as np


class OnlineLinearRegression:
    def __init__(self, n_dim, lambda_):
        self.n_dim=n_dim
        self.lambda_=lambda_
        self.A=np.identity(n=self.n_dim)#*1/(self.lambda_)
        self.b=np.zeros(self.n_dim)

    @property
    def inv_A(self):
        return np.linalg.inv(self.A)
    @property
    def w(self):
        return self.inv_A.dot(self.b)

    def fit(self, X, y):
        if len(X.shape)>1 and X.shape[1]==self.A.shape[0]:
            self.A=self.A+X.T.dot(X)
            self.b=self.b+X.T.dot(y)
        else:
            if X.shape[0]==self.A.shape[0]:
                X_=X.reshape(-1,1)
                self.A=self.A+X_.dot(X_.T)
            else:
                raise ValueError("X has not the expected shape, dimension should be {s} and it is {a}".format(s=self.A.shape[0], a=X.shape[0]))


    def predict(self, X):
        # X =(n,k) w=(k,)
        return X.dot(self.w)

    def return_param(self):
        return self.inv_A, self.w


class LinearUCB(ContextualBandit):
    def __init__(self, n_arms, context_dim, lambda_=1, alpha=0.5, chunks="auto"):
        super().__init__(n_arms, context_dim, chunks)
        self.lambda_=lambda_
        self.alpha=alpha
        self.models=[OnlineLinearRegression(self.context_dim, self.lambda_) for x in range(self.n_arms)]
        self.name="LinearUCB"

    def action(self, context):
        c=context.reshape(1,-1)
        bounds=[x.predict(c)+self.alpha*np.sqrt((c.dot(x.inv_A)).dot(c.T)) for x in self.models]
        return np.argmax(bounds)

    def update(self, action, context, reward):
        self.models[action].fit(context, reward)
        self.cum_rewards[action]+=reward
        self.arm_rounds[action]+=1
