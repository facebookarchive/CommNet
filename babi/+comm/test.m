% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

test_data = {};
test_data.story = test_story;
test_data.qstory = test_qstory;
test_data.questions = test_questions;
total_test_err = 0;
total_test_cost = 0;
total_test_num = 0;
for k = 1:floor(size(test_questions,2)/config.batch_size)
    batch = (1:config.batch_size) + (k-1) * config.batch_size;
    [l, e] = comm.train_batch(models, config, params, dict, test_data, batch, true);        
    total_test_cost = total_test_cost + l;
    total_test_err = total_test_err + e;
    total_test_num = total_test_num + config.batch_size;
end           

test_error = total_test_err/total_test_num;
disp(['test error: ', num2str(test_error)]);
