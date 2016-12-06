-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

function batch_init(size)
    local batch = {}
    for i = 1, size do
        batch[i] = new_game()
    end
    return batch
end

function batch_input(batch, active, t)
    if g_opts.encoder_lut then
        return batch_input_lut(batch, active, t)
    end
    active = active:view(#batch, g_opts.nagents)
    local input = torch.Tensor(#batch, g_opts.nagents, 2*g_opts.visibility+1, 2*g_opts.visibility+1, g_opts.nwords)
    input:fill(0)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                g:get_visible_state(input[i][a])
            end
        end
    end
    input = input:view(#batch * g_opts.nagents, -1)
    return input
end

function batch_input_lut(batch, active, t)
    active = active:view(#batch, g_opts.nagents)
    local input = torch.Tensor(#batch, g_opts.nagents, g_opts.encoder_lut_size)
    input:fill(g_opts.encoder_lut_nil)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                g:get_visible_state(input[i][a], true)
            end
        end
    end
    input = input:view(#batch * g_opts.nagents, -1)
    return input
end

function batch_act(batch, action, active)
    active = active:view(#batch, g_opts.nagents)
    action = action:view(#batch, g_opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                g:act(action[i][a])
            end
        end
    end
end

function batch_reset_comm(batch)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            for b = 1, g_opts.nagents do
                set_current_agent(g, b)
                g.agent.attr['talk_' .. a] = nil
            end
        end
    end
end

function batch_act_comm(batch, action, mask)
    action = action:view(#batch, g_opts.nagents)
    batch_reset_comm(batch)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            for b = 1, g_opts.nagents do
                if mask[i][a][b] > 0 then
                    set_current_agent(g, b)
                    g.agent.attr['talk_' .. a] = 'talk' .. action[i][a]
                end
            end
        end
    end
end

function batch_reward(batch, active, is_last)
    active = active:view(#batch, g_opts.nagents)
    local reward = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                reward[i][a] = g:get_reward(is_last)
            end
        end
    end
    return reward:view(-1)
end

function batch_terminal_reward(batch)
    local reward = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        if g.get_terminal_reward then
            for a = 1, g_opts.nagents do
                set_current_agent(g, a)
                reward[i][a] = g:get_terminal_reward()
            end
        end
    end
    return reward:view(-1)
end

function batch_update(batch)
    for i, g in pairs(batch) do
        g:update()
    end
end

function batch_active(batch)
    local active = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            set_current_agent(g, a)
            if g:is_active() then
                active[i][a] = 1
                -- this is little hacky
                if torch.type(g) == 'CombatGame' and g.agent.killed then
                    active[i][a] = 0
                end
            end
        end
    end
    return active:view(-1)
end

function batch_success(batch)
    local success = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        if g:is_success() then
            success[i] = 1
        end
    end
    return success
end
