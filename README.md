# mab-algos

Package that implements non-contextual and contextual multi-armed bandit models.

## Install

To run the package you need to install Poetry first. Simply type in your console:

`pip install --user poetry`

Once you're done run `poetry install` within your root project folder, and poetry will happily begin installing
dependencies and creating your virtual environment.

Run `poetry shell` to enter your virtual environment and `exit` to exit it.

## Usage

#### Non-contextual Multi-Armed Bandits

_mab_algos_ implements algorithms in the form of classes that have two main methods: _action_ and _update_.

Moreover, there are classes that allows to emulate multi-armed bandits scenarios where decisions have to be made online. These classes are _Simple_Sampler_ and _SimpleExperiment_.

## Example

```

# import mab-algos
from mab_algos.simple_bandit import UCB1, E_Greedy, Random, Greedy, Thompson_Sampling
from mab_algos.experiment import Simple_Sampler, SimpleExperiment

# set number of arms
n_arms=7

# Set models to run in experiment
list_algos={"UCB1":UCB1(n_arms=n_arms), "e_greedy":E_Greedy(n_arms=n_arms, epsilon=0.1),"Greedy":Greedy(n_arms=n_arms), "Random":Random(n_arms), "TS":Thompson_Sampling(n_arms=n_arms, lower_bound=0, upper_bound=1)}

# Specify Sampler
sampler=Simple_Sampler(n_arms)

# Specify experiment
experiment=SimpleExperiment(list_algos, sampler)

# run experiment
experiment.run(n_iter=1000)

# plot cummulative rewards
experiment.plot_cum_reward()

```

## Documentation
 In order to create documentation, from the _docs_ folder simply type `make html`. Documentation will be in `docs/_build`.
