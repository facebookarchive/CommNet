# Communication Neural Network (CommNet)
This is a set of codes used in our [paper](https://arxiv.org/abs/1605.07736) for training CommNet model, where neural network based agents learn to communicate for better cooperation.

The main code is for training CommNet and other baselines on traffic and combat tasks. The code in [babi](babi) subdirectory is for training the model on [bAbI tasks](http://fb.ai/babi), and its documentation can be found [here](babi/README.md). 
The code for the lever pulling task can be found in [levers](levers) subdirectory. The remainder of this documentation is for the main code.

## Requirements
All you need is [Torch7](http://torch.ch/) and its [nngraph](http://github.com/torch/nngraph) package. Optionally, install **ansicolors** for better visualization, and **gnuplot** for plotting during training.

    luarocks install ansicolors
    luarocks install gnuplot

The code only uses CPUs with multi-threading for speed-up.

## Usage
Running `th main.lua` will start training, but there are many command-line options. 

Among them, model related options:
```
  --model              module type: mlp | rnn | lstm [mlp]
  --nhop               the number of model steps per action [1]
  --hidsz              the size of the internal state vector [50]
  --nonlin             non-linearity type: tanh | relu | none [tanh]
  --init_std           STD of initial weights [0.2]
  --init_hid           initial value of internal state [0.1]
  --unshare_hops       not share weights of different hops [false]
  --encoder_lut        use LookupTable in encoder instead of Linear [false]
  --encoder_lut_size   max items in encoder LookupTable [50]
  --unroll             unroll steps for recurrent model. 0 means full unrolling. [10]
  --unroll_freq        unroll after every several steps [4]
  ```
Communication related options:
  ```
  --comm               enable continuous communication (CommNet) [false]
  --comm_mode          operation on incoming communication: avg | sum [avg]
  --comm_scale_div     divide comm vectors by this [1]
  --comm_encoder       encode incoming comm: 0=identity | 1=linear [0]
  --comm_decoder       decode outgoing comm: 0=identity | 1=linear | 2=nonlin [1]
  --comm_zero_init     initialize comm weights to zero [false]
  --comm_range         disable comm if L0 distance is greater than range [0]
  --nactions_comm      enable discrete communication when larger than 1 [1]
  --dcomm_entropy_cost entropy regularization for discrete communication [0]
  --fully_connected    use fully-connected model for all agents [false]
  ```
Game related options:
  ```
  --nagents            the number of agents [1]
  --nactions           the number of agent actions [6]
  --max_steps          force to end the game after this many steps [20]
  --games_config_path  configuration file for games [games/config/crossing.lua]
  --game               can specify a single game []
  --visibility         vision range of agents [1]
  ```
Training related options:
  ```
  --optim              optimization method: rmsprop | sgd | adam [rmsprop]
  --lrate              learning rate [0.001]
  --max_grad_norm      gradient clip value [0]
  --clip_grad          gradient clip value [0]
  --alpha              coefficient of baseline term in the cost function [0.03]
  --epochs             the number of training epochs [100]
  --nbatches           the number of mini-batches in one epoch [100]
  --batch_size         size of mini-batch (the number of parallel games) in each thread [16]
  --nworker            the number of threads used for training [18]
  --reward_mult        coeff to multiply reward for bprop [1]
  ```
Optimization related options:
  ```
  --momentum           momentum for SGD [0]
  --wdecay             weight decay for SGD [0]
  --rmsprop_alpha      parameter of RMSProp [0.97]
  --rmsprop_eps        parameter of RMSProp [1e-06]
  --adam_beta1         parameter of Adam [0.9]
  --adam_beta2         parameter of Adam [0.999]
  --adam_eps           parameter of Adam [1e-08]
  ```
Other options:
  ```
  --save               file name to save the model []
  --load               file name to load the model []
  --show               show progress [false]
  --no_coop            agents are NOT cooperative [false]
  --plot               plot average reward during training [false]
  --curriculum_sta     start making harder after this many epochs [0]
  --curriculum_end     when to make the game hardest [0]
```

### Selecting model type
By default, the code uses independent controllers for agents. To use different model:
- CommNet: use flag `--comm`. Optionally, limit communication range with `--comm_range 5`.
- Fully-connected:  use flag `--fully_connected`
- Discrete communication: set `--nactions_comm 10`, where 10 is the number of words to be used.

### Test play
After finished training (`th -i` flag will keep torch alive), you can see the agent playing by running
    
    test()

### Other tips
- Total number of games in a mini-batch is `nworker x batch_size`. In the paper, we trained with batch size of 288 by setting `--nworker 18 --batch_size 16`, but the same result can be obtained with fewer threads (although slower). For example, to use 4 threads set `--batch_size 72 --nworker 4`.
- Training is faster with `--encoder_lut` option when vision range is large, which uses a sparse representation of input. But remember to set `--encoder_lut_size` large enough.
- Recurrent models are unrolled limited steps for better stability, but can be fully unrolled by setting `--unroll 0`.

## Training script samples
Train LSTM CommNet on easy traffic task.

    th main.lua --lrate 0.003 --model lstm --games_config_path games/config/crossing_easy.lua \
    --nagents 5 --max_steps 20 --nactions 2 --curriculum_sta 10 --curriculum_end 40 --comm \
    --visibility 0 --show --epochs 50 --show

Train LSTM CommNet on crossing task

    th main.lua --lrate 0.003 --model lstm --games_config_path games/config/crossing.lua \
    --nagents 10 --max_steps 40 --nactions 2 --epochs 300 --curriculum_sta 100 --curriculum_end 200 \
    --comm --show

Training RNN CommNet with local connectivity on crossing hard task

    th main.lua --lrate 0.003 --model rnn --games_config_path games/config/crossing_hard.lua \
    --nagents 20 --max_steps 40 --nactions 2 --epochs 300 --curriculum_sta 100 --curriculum_end 200 \
    --comm --comm_range 5 --show

Train LSTM on combat tasks
    
    th main.lua --lrate 0.001 --model lstm --games_config_path games/config/combat_game.lua \
    --nagents 5 --max_steps 40 --epochs 300 --comm
