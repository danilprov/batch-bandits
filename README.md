# batch-bandits
Implementation of popular bandit algorithms in batch environments. 

## Overview

The repository provides an opportunuty to run simulations or replay logged datasets in _sequential batch_ manner -  sequential interaction with the environment when responses are grouped in batches and observed by the agent only at the end of each batch. Broadly speaking, sequential batch learning is a more generalized way of learning which covers both offline and online settings as special cases bringing together their advantages.

The project is built upon [RL-GLue](https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0) framework, which provides an interface to connect agents, environments, and experiment programs. Note, that [MAB/rl_glue.py](MAB/rl_glue.py) and [CMAB/rl_glue.py](CMAB/rl_glue.py) were adapted to make batch interaction possible.


## Framework

Two particularly useful versions of the [multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit#Contextual_bandit) are implemented: Stochastic Multi-Armed Bandit ([MAB](MAB)) and Contextual Multi-Armed Bandit ([CMAB](CMAB)).
The key feature of the project is that both versions support parameter `<batch_size>` - a certain period of time when the agent interacts with the environment "blindly". Despite the batch setting is a property of the environment, this limitation is considered from a policy perspective. With this, it is assumed that it is not the online agent who works with the batch environment, but the batch policy interacts with the online environment.
