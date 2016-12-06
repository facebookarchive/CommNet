% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

% Train CommNet model on a bAbi task

task = 1; % task ID
disp(['training on task ', num2str(task)]);
data_dir = 'data'; % path to data

% parse data
disp('parsing data ...');
f = dir(fullfile(data_dir,['qa',num2str(task),'_*_train.txt']));
data_path = {fullfile(data_dir,f(1).name)};
f = dir(fullfile(data_dir,['qa',num2str(task),'_*_test.txt']));
test_data_path = {fullfile(data_dir,f(1).name)};
dict = containers.Map;
dict('nil') = 1;
[story, questions,qstory] = babi.parseTask(data_path, dict, false);
[test_story, test_questions, test_qstory] = babi.parseTask(test_data_path, dict, false);

% configuration script
disp('model configuration ...');
config_babi;

disp('building model ...');
comm.model;

disp('start training ...');
comm.train;

disp('testing ...');
comm.test;
