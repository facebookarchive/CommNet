% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Weight < handle
    properties
        sz
        D
        grad
                
        mom
        v
        t
    end
    methods
        function obj = Weight(sz)
            obj.sz = sz;
            obj.D = 0.1 * randn(sz, 'single');
            obj.grad = zeros(sz, 'single');
            obj.mom = zeros(sz, 'single');
            obj.v = zeros(sz, 'single');
            obj.t = 1;
        end
        function update(obj, params)
            if isfield(params, 'set_init_std')
                obj.D = params.set_init_std * randn(obj.sz, 'single');
                return
            end
            if isfield(params, 'wc') && params.wc > 0
                obj.grad = obj.grad  + params.wc * obj.D;
            end
            if isfield(params, 'max_grad_norm') && params.max_grad_norm > 0
                if norm(obj.grad) > params.max_grad_norm
                    obj.grad = obj.grad * params.max_grad_norm / norm(obj.grad);
                end
            end
            
            if isfield(params, 'mom') && params.mom > 0
                obj.mom = params.mom * obj.mom + (1-params.mom) * obj.grad;
            else
                obj.mom = obj.grad;
            end
            
            if isfield(params, 'mom2') && params.mom2 > 0
                obj.v = params.mom2 * obj.v + (1-params.mom2) * (obj.grad.^2);
                denom = sqrt(obj.v) + 1e-6;
                corr1 = (1 - params.mom^obj.t);
                corr2 = (1 - params.mom2^obj.t);
                obj.D = obj.D - params.lrate * (sqrt(corr2)/corr1) * obj.mom ./ denom;
                obj.t = obj.t + 1;
            else                
                obj.D = obj.D - params.lrate * obj.mom;
            end
            
            if isfield(params, 'max_norm') && params.max_norm > 0
                dnorm = sqrt(sum(obj.D.^2,1));
                dnorm(dnorm < params.max_norm) = 1;
                obj.D = bsxfun(@rdivide, obj.D, dnorm);                
            end
            
            obj.grad(:) = 0;
        end
        function m = clone(obj)
            m = Weight(obj.sz);
            m.D = obj.D;
            m.grad = obj.grad;
        end
    end
end