# batch-bandits
Implementation of popular bandit algorithms in batch environments. 

Source code to our paper ["The Impact of Batch Learning in Stochastic Bandits"](https://arxiv.org/abs/2111.02071) accepted at the workshop on the [Ecological Theory of Reinforcement Learning](https://sites.google.com/view/ecorl2021/home?authuser=0), NeurIPS 2021.

## Overview

The repository provides an opportunuty to run simulations or replay logged datasets in _sequential batch_ manner -  sequential interaction with the environment when responses are grouped in batches and observed by the agent only at the end of each batch. Broadly speaking, sequential batch learning is a more generalized way of learning which covers both offline and online settings as special cases bringing together their advantages.


## Framework

Two particularly useful versions of the [multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit#Contextual_bandit) are implemented: Stochastic Multi-Armed Bandit ([MAB](MAB)) and Contextual Multi-Armed Bandit ([CMAB](CMAB)).
The key feature of the project is that both versions support parameter `batch_size` - a certain period of time when the agent interacts with the environment "blindly". Despite the batch setting is a property of the environment, this limitation is considered from a policy perspective. With this, it is assumed that it is not the online agent who works with the batch environment, but the batch policy interacts with the online environment.

The project is built upon [RL-GLue](https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0) framework, which provides an interface to connect agents, environments, and experiment programs. Note, that [MAB/rl_glue.py](MAB/rl_glue.py) and [CMAB/rl_glue.py](CMAB/rl_glue.py) were adapted to make batch interaction possible.

### Implemented algorithms

Version | Algorithm | Comment
------------ | ------------- | ------------- 
MAB | ε - greedy | -
MAB | Thompson Sampling | -
MAB | UCB | -
CMAB | LinTS | see [link](https://gdmarmerola.github.io/ts-for-contextual-bandits/) (and references therein) for more details
CMAB | LinUCB | see [article](https://arxiv.org/abs/1003.0146) for theoretical description
CMAB | Offline evaluator | policy evaluation technique; see [article](https://arxiv.org/abs/1003.5956) for theoretical quarantees
