% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

if config.position_enc == true
    % build fixed matrix for position-encoding 
    % see https://arxiv.org/abs/1503.08895 for more detail
    config.weight = ones(config.hidsz, config.max_words, 'single');
    for i = 1:config.hidsz
        for j = 1:config.max_words
            config.weight(i,j) = (i-(config.hidsz+1)/2)*(j-(config.max_words+1)/2);
        end
    end
    config.weight = 1 + 4 * config.weight / config.hidsz / config.max_words;
end

models = {};
models.lut_list = {};
lut1 = nn.LookUpTable(config.voc_sz, config.hidsz);
lut2 = nn.LookUpTable(config.voc_sz, config.hidsz);
models.lut_list{end+1} = lut1;
models.lut_list{end+1} = lut2;

% build encoder
enc = nn.Sequential();
P = nn.Parallel();
S1 = nn.Sequential();
S1.add(lut1);
if config.position_enc == true
    S1.add(nn.ElemMult(config.weight));
end
S1.add(nn.Sum(2));
S2 = nn.Sequential();
S2.add(lut2);
if config.position_enc == true
    S2.add(nn.ElemMult(config.weight));
end
S2.add(nn.Sum(2));
P.add(S1);
P.add(S2);
enc.add(P);
enc.add(nn.AddTable());
enc.add(nn.ReLU());
for i = 2:config.layers
    enc.add(nn.Linear(config.hidsz, config.hidsz));
    enc.add(nn.ReLU());
end
enc.add(nn.Duplicate());
models.enc = enc;

models.hops = {};
for h = 1:config.nhops
    % build a hop
    H = nn.Sequential();
    P = nn.Parallel();
    P.add(nn.Linear(config.hidsz, config.hidsz)); % for incoming communication
    P.add(nn.Linear(config.hidsz, config.hidsz)); % for internal state
    if config.input_persist
        % skip-connection
        S3 = nn.Sequential();
        S3.add(nn.LookUpTable(config.voc_sz, config.hidsz));
        if config.position_enc == true
            S3.add(nn.ElemMult(config.weight));
        end
        S3.add(nn.Sum(2));
        P.add(S3);
        % always share encoder LUT
        S3.modules{1}.share(lut2)
    end
    
    H.add(P);
    H.add(nn.AddTable());
    H.add(nn.ReLU());
    for i = 2:config.layers
        H.add(nn.Linear(config.hidsz, config.hidsz));
        H.add(nn.ReLU());
    end
    H.add(nn.Duplicate());

    if config.share
        if h > 1
            H.share(models.hops{1});
        end
    end
    models.hops{h} = H;
end

% decoder
dec = nn.Sequential();
dec.add(nn.Linear(config.hidsz, config.voc_sz));
dec.add(nn.Softmax());

% cost
loss = nn.CrossEntropyLoss();
loss.size_average = false;
loss.do_softmax_brop = true;
dec.modules{end}.skip_bprop = true;

models.dec = dec;
models.loss = loss;

% initialize params
params = {};
params.set_init_std = config.init_std;
models.enc.update(params);
for h = 1:config.nhops
    models.hops{h}.update(params);
end
models.dec.update(params);
