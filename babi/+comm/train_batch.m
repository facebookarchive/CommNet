% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

function [loss, error] = train_batch(models, config, params, dict, data, batch, test_only)
query = zeros(config.max_words, config.sz, config.batch_size,'single');
context = zeros(config.max_words, config.sz, config.batch_size,'single');
comm_graph = zeros(config.sz, config.batch_size, config.sz, config.batch_size,'single');
comm_graph_end = zeros(config.sz, config.batch_size, config.batch_size,'single');
target = data.questions(3,batch);

query(:) = dict('nil');
context(:) = dict('nil');
sz = zeros(config.batch_size,1);
for b = 1:config.batch_size
    % prepare a mini-batch
    d = data.story(:,1:data.questions(2,batch(b)),data.questions(1,batch(b)));
    offset = max(0,size(d,2)-config.sz);
    d = d(:,1+offset:end);
    sz(b) = size(d,2);
    context(1:size(d,1),1:sz(b),b) = d;
    if config.enable_time
        context(end,1:sz(b),b) = (sz(b):-1:1) + length(dict);
    end
    query(1:size(data.qstory,1),1:1:sz(b),b) = repmat(data.qstory(:,batch(b)), 1, sz(b));
    if sz(b) > 1
        % no self talking, and average communication vectors
        comm_graph(1:sz(b),b,1:sz(b),b) = (ones(sz(b)) - eye(sz(b)))/(sz(b)-1);
    end
    % sum final hidden states
    comm_graph_end(1:sz(b),b,b) = 1; 
end
comm_graph = reshape(comm_graph, config.sz * config.batch_size, []);
comm_graph_end = reshape(comm_graph_end, config.sz * config.batch_size, []);

comm_graph_hops = {};
for h = 1:config.nhops
    comm_graph_hops{h} = comm_graph;
end

query = reshape(query, size(query,1), []);
context = reshape(context, size(context,1), []);

% encoder forward
input_enc = {query, context};
out = models.enc.fprop(input_enc);

% forward through hops
input_hops = {};
for h = 1:config.nhops
    input_hops{h} = {out{1} * comm_graph_hops{h}}; % incoming communication
    input_hops{h}{2} = out{2}; % previous internal state
    if config.input_persist
        input_hops{h}{end+1} = context; % direct input
    end
    out = models.hops{h}.fprop(input_hops{h});
end

% decoder forward
input_dec = out{1} * comm_graph_end;
out = models.dec.fprop(input_dec);
loss = models.loss.fprop(out, target);
error = models.loss.get_error(out, target);

if test_only == false
    % back-propagation
    grad_dec = models.loss.bprop(out, target);
    grad_dec = models.dec.bprop(input_dec, grad_dec);
    grad = {};
    grad{1} = grad_dec * comm_graph_end';
    grad{2} = zeros(size(grad{1}));
    
    for h = config.nhops:-1:1
        gg = models.hops{h}.bprop(input_hops{h}, grad);    
        grad = {};
        grad{1} = gg{1} * comm_graph_hops{h}';
        grad{2} = gg{2};
    end
    
    models.enc.bprop(input_enc, grad);
    
    % update parameters
    models.enc.update(params);
    for h = 1:config.nhops
        models.hops{h}.update(params);
    end
    models.dec.update(params);
    % zero embedding of nil word
    for i = 1:length(models.lut_list)
        models.lut_list{i}.weight.D(:,dict('nil')) = 0;
    end
end
end