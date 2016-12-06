% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

params = {};
params.lrate = config.init_lrate;
params.max_grad_norm = config.max_grad_norm;
params.wc = config.wc;
params.mom = config.mom;
params.mom2 = config.mom2;
if isfield(config, 'max_norm')
    params.max_norm = config.max_norm;
end
train_data = {};
train_data.story = story;
train_data.qstory = qstory;
train_data.questions = questions;

for ep = 1:config.nepochs
    total_err = 0;
    total_cost = 0;
    total_num = 0;
    for k = 1:floor(length(train_range)/config.batch_size)
        batch = train_range(randi(length(train_range), config.batch_size,1));
        [l, e] = comm.train_batch(models, config, params, dict, train_data, batch, false);        
        total_cost = total_cost + l;
        total_err = total_err + e;
        total_num = total_num + config.batch_size;
    end           
    total_val_err = 0;
    total_val_cost = 0;
    total_val_num = 0;
    for k = 1:floor(length(val_range)/config.batch_size)
        % do validation
        batch = val_range((1:config.batch_size) + (k-1) * config.batch_size);
        [l, e] = comm.train_batch(models, config, params, dict, train_data, batch, true);        
        total_val_cost = total_val_cost + l;
        total_val_err = total_val_err + e;
        total_val_num = total_val_num + config.batch_size;
    end
    
    train_error = total_err/total_num;
    val_error = total_val_err/total_val_num;
    disp([num2str(ep), ' | train error: ', num2str(train_error), ' | val error: ', num2str(val_error)]);
end
