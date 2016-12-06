% Copyright (c) 2016-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef MSELoss < nn.Loss
    properties
        size_average = true;
    end
    methods
        function obj = MSELoss()
            obj = obj@nn.Loss();
        end
        function cost = fprop(obj, input, target)
            cost = (target - input).^2;
            cost = sum(cost(:));
            if obj.size_average
                cost = cost / size(input,2);
            end
        end
        function grad_input = bprop(obj, input, target)
            grad_input = 2 * (input - target);
            if obj.size_average
                grad_input = grad_input / size(input,2);
            end
        end
    end    
end