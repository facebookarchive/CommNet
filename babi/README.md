# CommNet model for bAbI tasks
This code is for training CommNet model on toy Q&A dataset [bAbI](http://fb.ai/babi), where one has to answer to a simple question after reading a short story. The model solves this problem by assigning each sentence of the story to separate agents, and let them communicate. After several steps of communication, agents produce a single answer. For more details, see our [paper](https://arxiv.org/abs/1605.07736).

## Usage
The code is written in Matlab. After downloading the code, go to the code directory and type `run` in Matlab. This will start training on the first task. To train on different task, change `task` variable in file `run.m`. The data directory contains older 10k version of tasks, but  the latest version can be downloaded from [here](http://fb.ai/babi).

You can change model settings in `config_babi.m`. With the default configuration, we obtained the following result, which included in the paper.

Task | Test error (%)
-----|---------:
1: 1 supporting fact |	0.00
2: 2 supporting facts |	3.23
3: 3 supporting facts |	68.35
4: 2 argument relations |	0.00
5: 3 argument relations |	1.71
6: yes/no questions |	0.00
7: counting	| 0.60
8: lists/sets	| 0.50
9: simple negation | 0.00
10: indefinite knowledge | 0.00
11: basic coherence	| 0.00
12: conjunction	| 0.00
13: compound coherence |0.00
14: time reasoning | 0.00
15: basic deduction	| 0.00
16: basic induction	| 51.31
17: positional reasoning | 15.12
18: size reasoning | 1.41
19: path finding | 0.00
20: agent's motivation | 0.00
Mean | 7.11
failed tasks | 3

