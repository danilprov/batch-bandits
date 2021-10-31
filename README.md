# batch-bandits
Implementation of popular bandit algorithms in batch environments. 

## Overview

The repository provides an opportunuty to run simulations or replay logged datasets in _sequential batch_ manner --  sequential interaction with the environment when responses are grouped in batches and observed by the agent only at the end of each batch. Broadly speaking, sequential batch learning is a more generalized way of learning which covers both offline and online settings as special casesbringing together their advantages.

The project is built upon [RL-GLue](https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0) framework, which provides an interface to connect agents, environments, and experiment programs. Note, that [MAB/rl_glue.py](MAB/rl_glue.py) and [CMAB/rl_glue.py](CMAB/rl_glue.py) were adapted to make batch interaction possible.


