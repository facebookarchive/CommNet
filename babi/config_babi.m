% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

config = {};
config.nhops = 3; % number of hops (including initial encoder)
config.layers = 2; % number of layers in each hop
config.batch_size = 32;
config.nepochs = 100;
config.hidsz = 100; % hidden state dimension
config.sz = min(50, size(story,2)); % limit number of sentences
config.voc_sz = length(dict);
config.position_enc = true; % use position-encoding instead of BoW
config.max_words = size(story,1);
config.enable_time = true; % add time embedings
if config.enable_time 
   config.voc_sz = config.voc_sz + config.sz;
   config.max_words = config.max_words + 1; % +1 for time words
end
config.input_persist = true; % skip-connection (each hop sees input)
config.share = false; % all hops share parameters

config.init_lrate = 0.003; % learning rate
config.max_grad_norm = 40; % clip large gradient
config.wc = 0.0; % weight cost regularization
config.mom = 0.9; % momentum (beta_1 of Adam)
config.mom2 = 0.99; % beta_2 parameter of Adam optimizer
config.init_std = 0.2; % initial weight std

% use 10% of training data for validation
train_range = 1:floor(0.9 * size(questions,2));
val_range = (floor(0.9 * size(questions,2))+1):size(questions,2);
