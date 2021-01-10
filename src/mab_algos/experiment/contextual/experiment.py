# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple example of contextual bandits simulation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import os, sys
import io
import matplotlib.pyplot as plt
from .helpers.simulator import run_contextual_bandit
from .helpers.data_sampler import sample_adult_data
from .helpers.data_sampler import sample_census_data
from .helpers.data_sampler import sample_covertype_data
from .helpers.data_sampler import sample_jester_data
from .helpers.data_sampler import sample_mushroom_data
from .helpers.data_sampler import sample_statlog_data
from .helpers.data_sampler import sample_stock_data

from .helpers.synthetic_data_sampler import sample_linear_data
from .helpers.synthetic_data_sampler import sample_sparse_linear_data
from .helpers.synthetic_data_sampler import sample_wheel_bandit_data
import pandas as pd

#added this for Jupyter noteboook
from .helpers.flags import Flags


class Hparams():
    def __init__(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)
    def add(self, dic):
        for key, value in dic.items():
            setattr(self, key, value)


def sample_data(data_type, num_contexts=None, directory=None):
  """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """

  FLAGS=Flags(directory=directory)

  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
  elif data_type == 'sparse_linear':
    # Create sparse linear dataset
    num_actions = 7
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    num_nnz_dims = int(context_dim / 3.0)
    dataset, _, opt_sparse_linear = sample_sparse_linear_data(
        num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
    opt_rewards, opt_actions = opt_sparse_linear
  elif data_type == 'mushroom':
    # Create mushroom dataset
    num_actions = 2
    context_dim = 117
    file_name = FLAGS.mushroom_data
    dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
    opt_rewards, opt_actions = opt_mushroom
  elif data_type == 'financial':
    num_actions = 8
    context_dim = 21
    num_contexts = min(3713, num_contexts)
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    file_name = FLAGS.financial_data
    dataset, opt_financial = sample_stock_data(file_name, context_dim,
                                               num_actions, num_contexts,
                                               noise_stds, shuffle_rows=True)
    opt_rewards, opt_actions = opt_financial
  elif data_type == 'jester':
    num_actions = 8
    context_dim = 32
    num_contexts = min(19181, num_contexts)
    file_name = FLAGS.jester_data
    dataset, opt_jester = sample_jester_data(file_name, context_dim,
                                             num_actions, num_contexts,
                                             shuffle_rows=True,
                                             shuffle_cols=True)
    opt_rewards, opt_actions = opt_jester
  elif data_type == 'statlog':
    file_name = FLAGS.statlog_data
    num_actions = 7
    num_contexts = min(43500, num_contexts)
    sampled_vals = sample_statlog_data(file_name, num_contexts,
                                       shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'adult':
    file_name = FLAGS.adult_data
    num_actions = 14
    num_contexts = min(45222, num_contexts)
    sampled_vals = sample_adult_data(file_name, num_contexts,
                                     shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'covertype':
    file_name = FLAGS.covertype_data
    num_actions = 7
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_covertype_data(file_name, num_contexts,
                                         shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'census':
    file_name = FLAGS.census_data
    num_actions = 9
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_census_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'wheel':
    delta = 0.95
    num_actions = 5
    context_dim = 2
    mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
    std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
    mu_large = 50
    std_large = 0.01
    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel

  return dataset, opt_rewards, opt_actions, num_actions, context_dim


def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def run_experiment(algos, data_type = 'mushroom',num_contexts = 2000, directory=None):

  # Problem parameters


  # Data type in {linear, sparse_linear, mushroom, financial, jester,
  #                 statlog, adult, covertype, census, wheel}

  # Create dataset

  #sys.__stdout__ = sys.stdout
  #sys.stdout = open(os.devnull, 'w')
  sampled_vals = sample_data(data_type, num_contexts, directory)
  dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals
  # Define hyperparameters and algorithms

  # Run contextual bandit problem
  t_init = time.time()
  results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
  #sys.stdout = sys.__stdout__
  h_actions, h_rewards = results
  return h_rewards, h_actions, algos, opt_rewards, opt_actions, t_init

class Contextual_Experiment:
    def __init__(self, data_type = 'mushroom',num_contexts = 2000, algos=None, directory=None):
        if data_type not in ["linear", "sparse_linear", "mushroom", "financial", "jester","statlog", "adult", "covertype", "census", "wheel"]:
            raise ValueError("datatype value must be one of the following: {linear, sparse_linear, mushroom, financial, jester, statlog, adult, covertype, census, wheel}")
        h_rewards,h_actions, algos, opt_rewards, opt_actions, t_init=run_experiment(data_type=data_type,num_contexts=num_contexts, algos=algos, directory=directory)

        self.data_type=data_type
        self.h_rewards=h_rewards
        self.h_actions=h_actions
        self.algos=algos
        self.opt_rewards=opt_rewards
        self.opt_actions=opt_actions
        self.t_init=t_init

    def show_analysis(self):
        # Display results
        display_results(self.algos, self.opt_rewards, self.opt_actions, self.h_rewards, self.t_init, self.data_type)

    def plot_cummulative_reward(self, steps=None):
        performance_pairs = []
        plt.figure(figsize=(12,10))
        if steps is None:
            for j, a in enumerate(self.algos):
                plt.plot(range(len(self.h_rewards)), np.cumsum(self.h_rewards[:, j]), label=a.name)
        else:
            for j, a in enumerate(self.algos):
                plt.plot(range(steps), np.cumsum(self.h_rewards[:steps, j]), label=a.name)
        plt.title("Comparisson Cummulative Reward")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Cummulative Reward")
        plt.show()

    def plot_cummulative_regreat(self, steps=None):
        performance_pairs = []
        plt.figure(figsize=(12,10))
        if steps is None:
            for j, a in enumerate(self.algos):
                plt.plot(range(len(self.h_rewards)), np.cumsum(self.opt_rewards-self.h_rewards[:,j]), label=a.name)
        else:
            for j, a in enumerate(self.algos):
                plt.plot(range(steps), np.cumsum(self.opt_rewards[:steps]-self.h_rewards[:steps,j]), label=a.name)
        plt.title("Comparisson Cummulative Regreat")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Cummulative Regreat")
        plt.show()



class Multiple_Contextual_Experiments(Contextual_Experiment):
    def __init__(self, num_runs=3, data_type = 'mushroom',num_contexts = 2000, algos=None, directory=None):
        if data_type not in ["linear", "sparse_linear", "mushroom", "financial", "jester","statlog", "adult", "covertype", "census", "wheel"]:
            raise ValueError("datatype value must be one of the following: {linear, sparse_linear, mushroom, financial, jester, statlog, adult, covertype, census, wheel}")

        self.total_rewards=[]
        self.total_actions=[]
        t_init = time.time()
        for i in range(num_runs):
            print("Round {s}/{f}".format(s=i+1, f=num_runs))
            h_rewards,h_actions, algos, opt_rewards, opt_actions, _ = run_experiment(data_type,num_contexts, algos)
            self.total_rewards.append(h_rewards)
            self.total_actions.append(h_actions)
        self.total_rewards=np.stack(self.total_rewards)
        self.total_actions=np.stack(self.total_actions)
        self.data_type=data_type
        self.h_rewards=self.total_rewards.mean(axis=0)
        self.h_actions=h_actions
        self.algos=algos
        self.opt_rewards=opt_rewards
        self.opt_actions=opt_actions
        self.t_init=t_init
        self.algo_names=[a.name for a in self.algos]
    def show_analysis(self):
        return pd.DataFrame(self.total_rewards.sum(axis=1), columns=self.algo_names).describe().T.sort_values(by="mean",ascending=False)
