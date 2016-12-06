-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- each agent should pull a different lever...
require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'

function get_reward(batchids,acts)
    local R = torch.zeros(batchids[-1])
    for s = 2, batchids:size(1) do
        local n = batchids[s] - batchids[s-1]
        local b = acts:sub(batchids[s-1]+1,batchids[s])
        local sb,sv = b:sort(1)
        sb = sb:squeeze()
        local c = sb[1]
        local r = 0
        for t = 2, n do
            if sb[t] ~= c then
                r = r + 1
                c = sb[t]
            end
        end
        R:sub(batchids[s-1]+1,batchids[s]):fill(r/(n-1))
    end
    return R
end


function make_batch(opts,T,Y)
    T[1]:uniform():mul(opts.nagents):ceil()
    local u = T[1]:float()
    local y = u:clone()
    for s = 2, batchids:size(1) do
        local n = batchids[s] - batchids[s-1]
        local tval, pre_tid = u:sub(batchids[s-1]+1,batchids[s]):sort(1)
        local _,tid = pre_tid:sort()
        y:sub(batchids[s-1]+1,batchids[s]):copy(tid)
    end
    Y:copy(y)
    return T,Y
end

local cmd = torch.CmdLine()
cmd:option('--apg', 10)  --models per game
cmd:option('--nagents', 500)  --total number of agents
cmd:option('--nlevers', 10) --number of levers
cmd:option('--maxiter', 1000000)
cmd:option('--hdim', 20)
cmd:option('--slayers_depth', 1)
cmd:option('--nlayer', 2)
cmd:option('--batchsize', 64)
cmd:option('--verbose', 500)
cmd:option('--lr', .05)
cmd:option('--reward_only', false) -- if false, use supervision instead of reward
cmd:option('--comm', false)
cmd:option('--equal_bags', true)
cmd:option('--anneal', 1000000)
cmd:option('--clip', -1)
cmd:option('--savedir','')
cmd:option('--save',false)
cmd:option('--recurrent',false)
cmd:option('--logpath', '')
opts = cmd:parse(arg or {})
opts.mdim = opts.hdim
print(opts)

batchids = torch.LongTensor(opts.batchsize+1)
batchids[1] = 0
for s = 1, opts.batchsize do
    if opts.equal_bags then
        batchids[s+1] = batchids[s] + opts.apg
    else
        batchids[s+1] = batchids[s] + torch.random(opts.apg-1)+1
    end
end
N = batchids[-1]
T = {}
T[1] = torch.rand(N):mul(opts.nagents):ceil()
T[2] = torch.zeros(N,opts.hdim)
T[3] = torch.zeros(N,opts.hdim)
T[4] = torch.ones(N)
T[5] = batchids:clone()
Y = torch.zeros(N)
for s =1, 5 do
    T[s] = T[s]:cuda()
end
Y = Y:cuda()


model = paths.dofile('conv_model.lua')

lgrad = torch.zeros(T[1]:size(1),opts.nlevers)
grad = lgrad:clone()
grad= grad:cuda()

crit = nn.ClassNLLCriterion()
crit = crit:cuda()

P, dF = model:getParameters()


er = 0
tr = 0
baseline = 0


logfile = opts.logpath .. 'levers_apg' .. opts.apg .. '_nagents' .. opts.nagents ..
    '_nlev' .. opts.nlevers .. '_hdim' .. opts.hdim ..
    '_nl' .. opts.nlayer .. '_lr' .. opts.lr .. '_RO' .. tostring(opts.reward_only) ..
    '_comm' .. tostring(opts.comm) .. '_clp' .. opts.clip .. '.txt'


if opts.logpath ~= '' then
    F = io.open(logfile,'w')
end

for s = 1, opts.maxiter do
    make_batch(opts,T,Y)
    model:zeroGradParameters()
    out = model:forward(T)
    acts = torch.multinomial(torch.exp(out),1):squeeze()
    acts = torch.round(acts)
    R = get_reward(batchids,acts:float())
    tr = tr + R:mean()
    baseline = .99*baseline + .01*R:mean()
    R:add(-baseline)
    R:mul(-1)
    if opts.reward_only then
        lgrad:zero()
        for t = 1, grad:size(1) do
            lgrad[t][acts[t]] = R[t]
        end
        grad:copy(lgrad)
        model:backward(T, grad)
    else
        er = er + crit:forward(out,Y)
        model:backward(T, crit:backward(out,Y))
    end
    step = (s - 1 + opts.anneal)/opts.anneal
    dt = math.max(opts.lr*(1/step), .00001*opts.lr)
    if opts.clip > 0 then
        if dF:norm() > opts.clip then dF:div(dF:norm()):mul(opts.clip) end
    end
    P:add(-dt,dF)

    if s% opts.verbose == 1 then
        if not opts.reward_only then
            print('iteration ' .. s .. ' dt ' .. dt .. ' crit ' .. er/opts.verbose .. ' reward ' .. tr/opts.verbose)
            er = 0
        else
            print('iteration ' .. s .. ' dt ' .. dt .. ' reward ' .. tr/opts.verbose)
        end
        if opts.logpath ~= '' then
            F:write(tr/opts.verbose .. '\n')
            F:flush()
        end
        tr = 0
        if opts.save then
            torch.save(opts.savedir .. 'model.th',model)
        end
    end
end
if opts.logpath ~= '' then
    F:close()
end
