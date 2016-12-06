-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('cutorch')
require('cunn')
require('cudnn')

-- builds a commnet with all-to-all communication.
-- uses a fixed input size for simplicity/efficiency:
-- if you know the number of agents in advance,
--  can implement with convolutions...
function build_streamlayer(opts)
    --inputs an
    -- 3 x opts.hdim x opts.apg*opts.batchsize x 1 tensor
    -- corresponding to initial mems, hidden, communication
    -- outputs an
    -- 2 x opts.hdim x opts.apg*opts.batchsize x 1 tensor
    -- corresponding to initial mems, hidden

    local P = nn.Sequential()
    P:add(nn.View(3*opts.hdim,opts.batchsize*opts.apg,1))
    P:add(cudnn.SpatialConvolution(3*opts.hdim,3*opts.hdim,1,1))
    for s = 1, opts.slayers_depth do
        P:add(nn.Threshold(0))
        P:add(cudnn.SpatialConvolution(3*opts.hdim,3*opts.hdim,1,1))
    end
    P:add(nn.Threshold(0))
    P:add(cudnn.SpatialConvolution(3*opts.hdim,opts.hdim,1,1))
    P:add(nn.View(1,opts.hdim,opts.batchsize*opts.apg,1))

    local Q = nn.Sequential()
    Q:add(nn.Select(1,1))
    Q:add(nn.View(1,opts.hdim,opts.batchsize*opts.apg,1))
    local slayer = nn.Concat(1)
    slayer:add(Q)
    slayer:add(P)
    return slayer
end

function build_communicate_layer(opts)
    local bs = opts.apg
    local h = opts.hdim

    local Q = nn.Identity()

    local A = nn.Sequential()
    A:add(nn.Select(1,2))
    if opts.communicate_rotate then
        A:add(cudnn.SpatialConvolution(h, h, 1, 1))
    end
    local K =cudnn.SpatialConvolution(h, h*bs, 1, bs, 1, bs)
    K.weight:zero()
    K.bias:zero()
    K.accGradParameters = function() end
    if opts.comm then
        for s = 1, bs do
            for t = 1, h do
                for u = 1, bs do
                    if s ~= u then K.weight[(t-1)*bs+s][t][u][1] = 1/(bs-1) end
                end
            end
        end
    end
    A:add(K)
    A:add(nn.View(h,bs,opts.batchsize,1));
    A:add(nn.Transpose({2,3}));
    A:add(nn.View(1,h,opts.batchsize*bs,1))
    A:add(nn.Contiguous())
    A:add(nn.View(1,opts.hdim,opts.batchsize*opts.apg,1))

    local C = nn.Concat(1)
    C:add(Q)
    C:add(A)
    return C
end


local model = nn.Sequential()

local baselayer = nn.ParallelTable()
mems = nn.Sequential();
mems:add(nn.LookupTable(opts.nagents + 1, opts.mdim))
mems:add(nn.Transpose({1,2}))
mems:add(nn.View(1,opts.mdim,opts.batchsize*opts.apg,1))
baselayer:add(mems)
baselayer:add(nn.View(1,opts.mdim,opts.batchsize*opts.apg,1))
baselayer:add(nn.View(1,opts.mdim,opts.batchsize*opts.apg,1))

model:add(baselayer)
model:add(nn.JoinTable(1))
local bs
for s = 1, opts.nlayer do
    if opts.recurrent then
        if s ==1 then
            bs = build_streamlayer(opts)
            model:add(bs)
        else
            local l = build_streamlayer(opts)
            l:share(bs,'weight','gradWeight','bias','gradBias')
            model:add(l)
        end
    else
        model:add(build_streamlayer(opts))
    end
    model:add(build_communicate_layer(opts))
end
if opts.recurrent then
    local l = build_streamlayer(opts)
    l:share(bs,'weight','gradWeight','bias','gradBias')
    model:add(l)
else
    model:add(build_streamlayer(opts))
end

model:add(nn.Select(1,2))
model:add(nn.View(opts.hdim, opts.batchsize*opts.apg))
model:add(nn.Transpose({1,2}))
model:add(nn.Linear(opts.hdim, opts.nlevers))
model:add(nn.LogSoftMax())
model:cuda()

return model
