-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('nngraph')
paths.dofile('linear_multi.lua')

local function nonlin()
    if g_opts.nonlin == 'tanh' then
        return nn.Tanh()
    elseif g_opts.nonlin == 'relu' then
        return nn.ReLU()
    elseif g_opts.nonlin == 'none' then
        return nn.Identity()
    else
        error('wrong nonlin')
    end
end

local function build_encoder(hidsz)
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    if g_opts.encoder_lut then
        in_dim = in_dim + 1 -- for NIL padding
        g_opts.encoder_lut_nil = in_dim
        local m = nn.LookupTable(in_dim, hidsz)
        g_modules['encoder_lut'] = m
        local s = nn.Sequential()
        s:add(m)
        s:add(nn.Sum(2))
        s:add(nn.Add(hidsz)) -- bias
        g_modules['encoder_sum'] = s.modules[3]
        return s
    else
        local m = nn.Linear(in_dim, hidsz)
        g_modules['encoder_linear'] = m
        return m
    end
end

local function build_rnn(input, prev_hid, comm_in, model_ids)
    local pre_hid = {}
    table.insert(pre_hid, build_encoder(g_opts.hidsz)(input))
    table.insert(pre_hid, linear_multi(g_opts.hidsz, g_opts.hidsz, model_ids, prev_hid))
    g_modules['pre_hid'] = pre_hid[2].data.module
    if comm_in then table.insert(pre_hid, comm_in) end
    local hidstate = nonlin()(nn.CAddTable()(pre_hid))
    return hidstate
end

local function build_lstm(input, prev_hid, prev_cell, comm_in, model_ids)
    local pre_hid = {}
    table.insert(pre_hid, build_encoder(g_opts.hidsz * 4)(input))
    table.insert(pre_hid, linear_multi(g_opts.hidsz, g_opts.hidsz * 4, model_ids, prev_hid))
    if comm_in then table.insert(pre_hid, comm_in) end
    local A = nn.CAddTable()(pre_hid)
    local B = nn.View(4, g_opts.hidsz):setNumInputDims(2)(A)
    local C = nn.SplitTable(1, 2)(B)

    local gate_forget = nn.Sigmoid()(nn.SelectTable(1)(C))
    local gate_write = nn.Sigmoid()(nn.SelectTable(2)(C))
    local gate_read = nn.Sigmoid()(nn.SelectTable(3)(C))
    local in2c = nonlin()(nn.SelectTable(4)(C))
    local cellstate = nn.CAddTable()({
        nn.CMulTable()({prev_cell, gate_forget}),
        nn.CMulTable()({in2c, gate_write})
        })
    local hidstate = nn.CMulTable()({nonlin()(cellstate), gate_read})
    return hidstate, cellstate
end

function g_build_model()
    g_model_inputs = {}
    g_model_outputs = {}
    local in_mods = {}
    local out_mods = {}

    local prev_hid = nn.Identity()()
    g_modules['prev_hid'] = prev_hid.data.module
    local input = nn.Identity()()
    table.insert(in_mods, input)
    g_model_inputs['input'] = #in_mods
    table.insert(in_mods, prev_hid)
    g_model_inputs['prev_hid'] = #in_mods

    local model_ids = nn.Identity()()
    if g_opts.nmodels > 1 then
        table.insert(in_mods, model_ids)
        g_model_inputs['model_ids'] = #in_mods
    end

    local comm2hid
    if g_opts.comm then
        local comm_in = nn.Identity()()
        table.insert(in_mods, comm_in)
        g_model_inputs['comm_in'] = #in_mods
        g_modules['comm_in'] = comm_in.data.module
        comm2hid = nn.Sum(2)(comm_in)
        if g_opts.comm_encoder == 1 then
            if g_opts.model == 'lstm' then
                comm2hid = linear_multi(g_opts.hidsz, g_opts.hidsz * 4, model_ids, comm2hid)
            else
                comm2hid = linear_multi(g_opts.hidsz, g_opts.hidsz, model_ids, comm2hid)
            end
            g_modules['comm_encoder'] = comm2hid
        end
    end

    local hidstate
    if g_opts.model == 'mlp' or g_opts.model == 'rnn' then
        hidstate = build_rnn(input, prev_hid, comm2hid, model_ids)
    elseif g_opts.model == 'lstm' then
        local prev_cell = nn.Identity()()
        g_modules['prev_cell'] = prev_cell.data.module
        table.insert(in_mods, prev_cell)
        g_model_inputs['prev_cell'] = #in_mods
        local cellstate
        hidstate, cellstate = build_lstm(input, prev_hid, prev_cell, comm2hid, model_ids)
        table.insert(out_mods, cellstate)
        g_model_outputs['cellstate'] = #out_mods
    else
        error('model not supported')
    end

    local action = linear_multi(g_opts.hidsz, g_opts.nactions, model_ids, hidstate)
    local action_prob = nn.LogSoftMax()(action)
    local baseline = linear_multi(g_opts.hidsz, 1, model_ids, hidstate)
    table.insert(out_mods, action_prob)
    g_model_outputs['action_prob'] = #out_mods
    table.insert(out_mods, baseline)
    g_model_outputs['baseline'] = #out_mods
    table.insert(out_mods, hidstate)
    g_model_outputs['hidstate'] = #out_mods

    if g_opts.comm then
        local comm_out
        if g_opts.fully_connected then
            -- use different params depending on agent ID
            comm_out =  linear_multi(g_opts.hidsz, g_opts.hidsz * g_opts.nagents, model_ids, hidstate)            
        else
            comm_out = hidstate
            if g_opts.comm_decoder >= 1 then
                comm_out =  linear_multi(g_opts.hidsz, g_opts.hidsz, model_ids, comm_out)
                g_modules['comm_decoder'] = comm_out
                if g_opts.comm_decoder == 2 then
                    comm_out = nonlin()(comm_out)
                end
            end
            comm_out = nn.Contiguous()(nn.Replicate(g_opts.nagents, 2)(comm_out))
        end
        table.insert(out_mods, comm_out)
        g_model_outputs['comm_out'] = #out_mods
    end

    if g_opts.nactions_comm > 1 then
        local action_comm = linear_multi(g_opts.hidsz, g_opts.nactions_comm, model_ids, hidstate)
        action_comm = nn.LogSoftMax()(action_comm)
        table.insert(out_mods, action_comm)
        g_model_outputs['action_comm'] = #out_mods
    end

    local model = nn.gModule(in_mods, out_mods)
    return model
end

function g_init_model()
    g_modules = {}
    g_model = g_build_model()
    g_paramx, g_paramdx = g_model:getParameters()
    if g_opts.init_std > 0 then
        g_paramx:normal(0, g_opts.init_std)
    end
    if g_opts.comm_zero_init then
        if g_opts.nmodels == 1 then
            if g_modules['comm_encoder'] then
                g_modules['comm_encoder'].data.module.weight:zero()
                g_modules['comm_encoder'].data.module.bias:zero()
            end
            if g_modules['comm_decoder'] then
                g_modules['comm_decoder'].data.module.weight:zero()
                g_modules['comm_decoder'].data.module.bias:zero()
            end
        else
            if g_modules['comm_encoder'] then
                g_modules['comm_encoder'].weight_lut.data.module.weight:zero()
                g_modules['comm_encoder'].bias_lut.data.module.weight:zero()
            end
            if g_modules['comm_decoder'] then
                g_modules['comm_decoder'].weight_lut.data.module.weight:zero()
                g_modules['comm_decoder'].bias_lut.data.module.weight:zero()
            end
        end
    end
    if g_opts.encoder_lut then
        -- zero NIL embedding
        g_modules['encoder_lut'].weight[g_opts.encoder_lut_nil]:zero()
    end
    g_bl_loss = nn.MSECriterion()
    g_bl_loss.sizeAverage = false
end
