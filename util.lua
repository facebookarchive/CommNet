-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function get_agent(g, a)
    if torch.type(g) == 'CombatGame' then
        local c = 0
        for i = 1, g.nagents do
            if (not g.agents[i].team) or g.agents[i].team == 'team1' then
                c = c + 1
                if c == a then
                    return g.agents[i]
                end
            end
        end
        error('can not find agent ' .. a)
    else
        return g.agents[a]
    end
end

function set_current_agent(g, a)
    g.agent = get_agent(g, a)
end

function merge_stat(stat, s)
    for k, v in pairs(s) do
        if type(v) == 'number' then
            stat[k] = (stat[k] or 0) + v
        elseif type(v) == 'table' then
            if v.op == 'join' then
                if stat[k] then
                    local sz = stat[k].data:size()
                    sz[1] = sz[1] + v.data:size(1)
                    stat[k].data:resize(sz)
                    stat[k].data:narrow(1, sz[1]-v.data:size(1)+1, v.data:size(1)):copy(v.data)
                else
                    stat[k] = {data = v.data:clone()}
                end
            end
        else
            -- it must be tensor
            if stat[k] then
                stat[k]:add(v)
            else
                stat[k] = v:clone()
            end
        end
    end
end

function sample_multinomial(p)
    -- for some reason multinomial fails sometimes
    local s, sample = pcall(
        function() 
            return torch.multinomial(p, 1) 
        end) 
    if s == false then
        sample = torch.multinomial(torch.ones(p:size()),1)
    end
    return sample
end

function tensor_to_words(input, show_prob)
    for i = 1, input:size(1) do
        local line = i .. ':'
        for j = 1, input:size(2) do
            line = line .. '\t'  .. g_ivocab[input[i][j]]
        end
        if show_prob then
            for h = 1, g_opts.nhop do
                line = line .. '\t' .. string.format('%.2f', g_modules[h]['prob'].output[1][i])
            end
        end
        print(line)
    end
end


function format_stat(stat)
    local a = {}
    for n in pairs(stat) do table.insert(a, n) end
    table.sort(a)
    local str = ''
    for i,n in ipairs(a) do
        if string.find(n,'count_') then
            str = str .. n .. ': ' .. string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    for i,n in ipairs(a) do
        if string.find(n,'reward_') then
            str = str .. n .. ': ' ..  string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    for i,n in ipairs(a) do
        if string.find(n,'success_') then
            str = str .. n .. ': ' ..  string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    str = str .. 'bl_cost: ' .. string.format("%2.4g",stat['bl_cost']) .. ' '
    str = str .. 'reward: ' .. string.format("%2.4g",stat['reward']) .. ' '
    str = str .. 'success: ' .. string.format("%2.4g",stat['success']) .. ' '
    str = str .. 'active: ' .. string.format("%2.4g",stat['step_active']) .. ' '
    str = str .. 'epoch: ' .. stat['epoch']
    return str
end
function print_tensor(a)
    local str = ''
    for s = 1, a:size(1) do str = str .. string.format("%2.4g",a[s]) .. ' '  end
    return str
end
function format_helpers(gname)
    local str = ''
    if not gname then
        for i,j in pairs(g_factory.helpers) do
            str = str .. i .. ' :: '
            str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
            str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
            str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
            str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
            str = str .. '\n'
        end
    else
        local j = g_factory.helpers[gname]
        str = str .. gname .. ' :: '
        str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
        str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
        str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
        str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
        str = str .. '\n'
    end
    return str
end

function g_load_model()
    if g_opts.load ~= '' then
        if paths.filep(g_opts.load) == false then
            print('WARNING: Failed to load from ' .. g_opts.load)
            return
        end
        local f = torch.load(g_opts.load)
        g_paramx:copy(f.paramx)
        g_log = f.log
        g_plot_stat = {}
        for i = 1, #g_log do
            g_plot_stat[i] = {g_log[i].epoch, g_log[i].reward, g_log[i].success, g_log[i].bl_cost}
        end
        if f['optim_state'] then g_optim_state = f['optim_state'] end
        print('model loaded from ', g_opts.load)
    end
end

function g_save_model()
    if g_opts.save ~= '' then
        f = {opts=g_opts, paramx=g_paramx, log=g_log}
        if g_optim_state then f['optim_state'] = g_optim_state end
        torch.save(g_opts.save, f)
        print('model saved to ', g_opts.save)
    end
end

function plot_reward()
    local x = torch.zeros(#g_log)
    for i = 1, #g_log do
        x[i] = g_log[i].reward
    end
    gnuplot.plot(x)
end