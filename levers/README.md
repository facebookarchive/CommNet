# Lever pulling task

This is a code used for training CommNet model on a lever pulling task, which requires the agents to communicate in order to win. 
This consists of m levers and a pool of N agents. At each round, m agents are drawn at random from the total pool of N 
agents and they must each choose a lever to pull, simultaneously with the other m − 1 agents, after which the round ends. 
The goal is for each of them to pull a different lever. Correspondingly, all agents receive reward proportional to the 
number of distinct levers pulled. Each agent can see its own identity, and nothing else.
For more details about the task, see our [paper](https://arxiv.org/abs/1605.07736).

## Usage
The code uses Torch and run on a GPU for speed-up. To start training, run

    th lever.lua --reward_only --comm
    
Here, option `--comm` enables a communication between the agents, and `--reward_only` picks a reinforcement learning instead of supervised learning.

The following script can be used to replicated the experiments in the paper.

    th lever.lua --batchsize 512 --lr 10 --clip .01 --hdim 64 --apg 5 --nlevers 5 --reward_only --maxiter 100000 --comm
