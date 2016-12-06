% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef VecScalMult < nn.Module
    properties
        c
    end
    methods
        function obj = VecScalMult()
            obj = obj@nn.Module();
        end
        function output = fprop(obj, input)
            obj.output = bsxfun(@times, input{1}, input{2});
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = {};
            obj.grad_input{1} = bsxfun(@times, grad_output, input{2});
            obj.grad_input{2} = sum(grad_output .* input{1}, 1);
            grad_input = obj.grad_input;
        end
    end
end