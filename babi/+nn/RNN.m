% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef RNN < nn.Module
    properties
        dict_sz;
        hid_dim;
        bprop_step;
        enc;
        dec;
        hist;
        init_val = 0.1;
        valid_steps;
    end
    methods
        function obj = RNN(config)
            obj = obj@nn.Module();
            obj.hid_dim = config.hid_dim;
            obj.dict_sz = config.dict_sz;
            obj.bprop_step = config.bprop_step;
            
            % encoder
            obj.hist = {};
            for i = 1:obj.bprop_step
                obj.hist{i}.enc = obj.constructEncoder(config);
                if i > 1
                    obj.hist{i}.enc.share(obj.hist{i-1}.enc);
                end
            end
            obj.enc = obj.hist{1}.enc;
            
            % decoder            
            obj.dec = obj.constructDecoder(config);
            
            obj.reset();
        end
        function enc = constructEncoder(obj, config)
            enc = nn.Sequential();
            m1 = nn.Parallel();
            m1.add(nn.LookUpTable(obj.dict_sz, obj.hid_dim))
            m1.add(nn.LinearNB(obj.hid_dim, obj.hid_dim))
            enc.add(m1)
            enc.add(nn.AddTable())
            enc.add(nn.Sigmoid())
        end
        function dec = constructDecoder(obj, config)
            dec = nn.Sequential();
            dec.add(nn.LinearNB(obj.hid_dim, obj.dict_sz));
            dec.add(nn.Softmax());
        end
        function reset(obj)
            obj.valid_steps = 0;
        end
        function output = fprop(obj, input)
            % shift history by one step
            curr = obj.hist{end};
            for i = length(obj.hist):-1:2
                obj.hist{i} = obj.hist{i-1};
            end
            obj.hist{1} = curr;

            obj.enc = obj.hist{1}.enc;            
            obj.hist{1}.input = input;
            obj.hist{1}.dec_grad = [];
            if obj.valid_steps == 0 
                obj.hist{1}.enc_input = {input, obj.init_val * ones(obj.hid_dim, size(input,2), 'single')};
            else
                obj.hist{1}.enc_input = {input, obj.hist{2}.hid_state};
            end
            obj.hist{1}.hid_state = obj.enc.fprop(obj.hist{1}.enc_input);
            obj.output = obj.dec.fprop(obj.hist{1}.hid_state);
            
            obj.valid_steps = obj.valid_steps + 1;
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            % only do partial bprop on decoder
            obj.hist{1}.dec_grad = obj.dec.bprop(obj.hist{1}.hid_state, grad_output);
            obj.grad_input = [];
            grad_input = obj.grad_input;
        end
        function update(obj, params)  
            % complete bprop-time on encoder
            grad = zeros(obj.hid_dim, size(obj.output,2), 'single');
            for i = 1:length(obj.hist)
                if i > obj.valid_steps
                    break
                end
                if isempty(obj.hist{i}.dec_grad) == false
                    grad = grad + obj.hist{i}.dec_grad;
                end
                g = obj.hist{i}.enc.bprop(obj.hist{i}.enc_input, grad);
                grad = g{2};
            end
            
            obj.enc.update(params);
            obj.dec.update(params);
        end
    end
end