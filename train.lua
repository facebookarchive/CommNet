-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require'optim'

function train_batch(test_run)
    -- start a new episode
    local stat = {}
    local batch = batch_init(g_opts.batch_size)

    -- TESTS
    assert(g_opts.nactions == batch[1].agent.nactions)
    local game_nagents = 0
    for i = 1, batch[1].nagents do
        if (not batch[1].agents[i].team) or batch[1].agents[i].team == 'team1' then
            game_nagents = game_nagents + 1
        end
    end
    assert(g_opts.nagents == game_nagents)

    -- record episode states
    local reward = {}
    local input = {}
    local action = {}
    local action_comm = {}
    local active = {}

    -- for recurrent models
    local in_dim = (g_opts.visibility*2+1)^2 * g_opts.nwords
    local hid_state = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.hidsz):fill(0)
    local hid_grad = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.hidsz):fill(0)
    local cell_state, cell_grad
    if g_opts.model == 'lstm' then
        cell_state = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.hidsz):fill(0)
        cell_grad = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.hidsz):fill(0)
    end
    if g_opts.recurrent then
        hid_state:fill(g_opts.init_hid)
        if g_opts.model == 'lstm' then
            cell_state:fill(g_opts.init_hid)
        end
    end

    -- communication states
    local comm_state = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
    local comm_grad = torch.zeros(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
    local comm_mask_default = torch.ones(g_opts.nagents, g_opts.nagents)
    local comm_mask = {}
    for s = 1, g_opts.nagents do
        for d = 1, g_opts.nagents do
            if s == d then
                -- no self talking
                comm_mask_default[s][d] = 0
            end
        end
    end

    local model_ids = {}
    local minds = torch.Tensor(#batch, g_opts.nagents)
    for b = 1, #batch do
        for a = 1, g_opts.nagents do
            if g_opts.fully_connected then
                minds[b][a] = a
            else
                minds[b][a] = torch.random(g_opts.nminds)
            end
        end
    end

    minds = minds:view(-1)
    for i = 1, g_opts.nhop do
        model_ids[i] = minds:clone()
        if g_opts.unshare_hops then
            model_ids[i]:add((i-1)*g_opts.nminds)
        end
    end

    -- play the games
    for t = 1, g_opts.max_steps * g_opts.nhop do
        active[t] = batch_active(batch)
        local x = batch_input(batch, active[t], t)

        input[t] = {}
        input[t][g_model_inputs['input']] = x
        local hop_id = (t-1) % g_opts.nhop + 1
        if g_opts.nmodels > 1 then
            input[t][g_model_inputs['model_ids']] = model_ids[hop_id]
        end
        input[t][g_model_inputs['prev_hid']] = hid_state
        if g_opts.model == 'lstm' then
            input[t][g_model_inputs['prev_cell']] = cell_state
        end
        if g_opts.comm then
            input[t][g_model_inputs['comm_in']] = comm_state
        end
        local out = g_model:forward(input[t])

        -- act when model hops completed
        if t % g_opts.nhop == 0 then
            stat.step_active = (stat.step_active or 0) + active[t]:sum()
            stat.step_count = (stat.step_count or 0) + #batch * g_opts.nagents
            action[t] = sample_multinomial(torch.exp(out[g_model_outputs['action_prob']]))
            if test_run then
                batch[1].map:print_ascii()
                os.execute('sleep 0.2')
            end
            batch_act(batch, action[t]:view(-1), active[t])
            batch_update(batch)
            reward[t] = batch_reward(batch, active[t], t == g_opts.max_steps * g_opts.nhop)
            if test_run then
                for a = 1, g_opts.nagents do
                    if active[t][a] == 1 then
                        local agent = get_agent(batch[1], a)
                        print('agent ' .. a .. ' reward:', reward[t][a],
                            'action: ' .. agent.action_names[action[t][a][1]],
                                'HP: ' .. (agent.attr.health or 0),
                                'name: ' .. agent.attr.name)
                    end
                end
            end
        end

        hid_state = out[g_model_outputs['hidstate']]:clone()
        if g_opts.model == 'lstm' then
            cell_state = out[g_model_outputs['cellstate']]:clone()
        end

        if g_opts.nactions_comm > 1 then
            -- discrete communication words
            action_comm[t] = sample_multinomial(torch.exp(out[g_model_outputs['action_comm']]))
        end

        if g_opts.comm or g_opts.nactions_comm > 1 then
            -- determine which agent can talk to which agent?
            local m = comm_mask_default:view(1, g_opts.nagents, g_opts.nagents)
            m = m:expand(g_opts.batch_size, g_opts.nagents, g_opts.nagents):clone()

            if g_opts.fully_connected then
                -- pass all comm because it is fully connected
            else
                -- inactive agents don't communicate
                local m2 = active[t]:view(g_opts.batch_size, g_opts.nagents, 1):clone()
                m2 = m2:expandAs(m):clone()
                m:cmul(m2)
                m:cmul(m2:transpose(2,3))
            end

            if g_opts.comm_range > 0 then
                -- far away agents can't communicate
                for i, g in pairs(batch) do
                    for s = 1, g_opts.nagents do
                        for d = 1, g_opts.nagents do
                            local dy = math.abs(get_agent(g, s).loc.y - get_agent(g, d).loc.y)
                            local dx = math.abs(get_agent(g, s).loc.x - get_agent(g, d).loc.x)
                            local r = math.max(dy, dx)
                            if r > g_opts.comm_range then
                                m[i][s][d] = 0
                            end
                        end
                    end
                end
            end

            if g_opts.comm_mode == 'avg' then
                -- average comms by dividing by number of agents
                m:cdiv(m:sum(2):expandAs(m):clone():add(m:eq(0):float()))
            end
            m:div(g_opts.comm_scale_div)
            comm_mask[t] = m
        end

        if g_opts.comm then
            -- communication vectors for next step
            local h = out[g_model_outputs['comm_out']]:clone()
            h = h:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, g_opts.hidsz)
            -- apply mask
            local m = comm_mask[t]
            m = m:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, 1)
            m = m:expandAs(h):clone()
            h:cmul(m)
            comm_state = h:transpose(2,3):clone()
            comm_state:resize(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
        end

        -- pass discrete communication words between agents
        -- (talk even between hops)
        if g_opts.nactions_comm > 1 then
            batch_act_comm(batch, action_comm[t]:view(-1), comm_mask[t])
            if test_run then
                for a = 1, g_opts.nagents do
                    print('agent ' .. a .. ' talk:' .. action_comm[t][a][1])
                end
            end
        end

        -- restart internal states if necessary
        if t % g_opts.nhop == 0 and g_opts.recurrent == false then
            hid_state:zero()
            if g_opts.model == 'lstm' then
                cell_state:zero()
            end
            if g_opts.comm then
                comm_state:zero()
            end
            if g_opts.nactions_comm > 1 then
                batch_reset_comm(batch)
            end
        end
    end

    local success = batch_success(batch)
    local reward_terminal = batch_terminal_reward(batch)
    if test_run then
        for a = 1, g_opts.nagents do
            print('agent' .. a, 'reward terminal: ' .. reward_terminal[a])
        end
        print('success: ' .. success[1])
        return
    end

    -- do back-propagation
    g_paramdx:zero()
    local reward_sum = torch.Tensor(g_opts.batch_size * g_opts.nagents):zero()
    reward_sum:add(reward_terminal)
    local t_step = 1
    if g_opts.unroll > 0 then t_step = g_opts.unroll_freq end
    -- go back in time
    for t_game = g_opts.max_steps, 1, -t_step do
        if g_opts.recurrent == false or g_opts.unroll > 0 then
            hid_grad:zero()
            if g_opts.model == 'lstm' then
                cell_grad:zero()
            end
            if g_opts.comm then
                comm_grad:zero()
            end
        end

        local t_sta = t_game * g_opts.nhop
        local t_end = (t_game - 1) * g_opts.nhop + 1
        if g_opts.unroll > 0 then
            t_end = t_end - g_opts.unroll * g_opts.nhop
        end
        t_end = math.max(1, t_end)
        for t = t_sta, t_end, -1  do
            local allow_reward_grad = true
            if t <= (t_game - t_step) * g_opts.nhop then allow_reward_grad = false end

            local out = g_model:forward(input[t])
            local grad = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.nactions):zero()
            local bl_grad = torch.Tensor(g_opts.batch_size * g_opts.nagents, 1):zero()
            local action_comm_grad
            if g_opts.nactions_comm > 1 then
                action_comm_grad = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.nactions_comm):zero()
            end
            if allow_reward_grad and t % g_opts.nhop == 0 then
                reward_sum:add(reward[t]) -- cumulative reward
            end

            local R = reward_sum:clone()

            if not g_opts.no_coop then
                -- average agents' rewards
                R = R:view(#batch, g_opts.nagents)
                R = R:mean(2):expandAs(R):clone()
                R = R:view(-1, 1)
            end
            R:cmul(active[t])
            R:mul(g_opts.reward_mult)

            local baseline
            if allow_reward_grad and (t % g_opts.nhop == 0 or g_opts.nactions_comm > 1) then
                -- do baseline update on every hop with discrete comm
                baseline = out[g_model_outputs['baseline']]
                if g_opts.bl_off then baseline:zero() end
                baseline:cmul(active[t])
                stat.bl_cost = (stat.bl_cost or 0) + g_bl_loss:forward(baseline, R)
                stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
                bl_grad = g_bl_loss:backward(baseline, R)
                bl_grad:mul(g_opts.alpha):div(g_opts.batch_size)
                if g_opts.bl_off then bl_grad:zero() end

                if g_opts.nactions_comm > 1 then
                    -- gradient for discrete comm words
                    action_comm_grad:scatter(2, action_comm[t], baseline - R)
                    if g_opts.dcomm_entropy_cost > 0 then
                        local logp = out[g_model_outputs['action_comm']]
                        local entropy_grad = logp:clone():add(1)
                        entropy_grad:cmul(torch.exp(logp))
                        entropy_grad:mul(g_opts.dcomm_entropy_cost)
                        entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
                        action_comm_grad:add(entropy_grad)
                    end
                    action_comm_grad:div(g_opts.batch_size)
                end
            end

            if allow_reward_grad and t % g_opts.nhop == 0 then
                local R_action = baseline - R
                grad:scatter(2, action[t], R_action)
                grad:div(g_opts.batch_size)
            end
            local all_grad = {}
            all_grad[g_model_outputs['action_prob']] = grad
            all_grad[g_model_outputs['baseline']] = bl_grad
            if g_opts.nactions_comm > 1 then
                all_grad[g_model_outputs['action_comm']] = action_comm_grad
            end
            all_grad[g_model_outputs['hidstate']] = hid_grad
            if g_opts.model == 'lstm' then
                all_grad[g_model_outputs['cellstate']] = cell_grad
            end
            if g_opts.comm then
                all_grad[g_model_outputs['comm_out']] = comm_grad
            end

            if g_opts.clip_grad > 0 then
                all_grad[g_model_outputs['hidstate']]:renorm(2, 1, g_opts.clip_grad)
                if g_opts.model == 'lstm' then
                    all_grad[g_model_outputs['cellstate']]:renorm(2, 1, g_opts.clip_grad)
                end
                if g_opts.comm then
                    all_grad[g_model_outputs['comm_out']]:renorm(2, 1, g_opts.clip_grad)
                end
            end

            g_model:backward(input[t], all_grad)
            hid_grad = g_modules['prev_hid'].gradInput:clone()
            if g_opts.model == 'lstm' then
                cell_grad = g_modules['prev_cell'].gradInput:clone()
            end

            if g_opts.comm and t > 1 then
                local h = g_modules['comm_in'].gradInput:clone()
                h = h:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, g_opts.hidsz)
                comm_grad = h:transpose(2,3):clone()
                -- apply mask
                local m = comm_mask[t-1]
                m = m:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, 1)
                m = m:expandAs(comm_grad):clone()
                comm_grad:cmul(m)
                comm_grad:resize(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
            end
        end
    end

    -- R:resize(g_opts.batch_size, g_opts.nagents)
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end

function train_batch_thread(opts_orig, paramx_orig)
    g_opts = opts_orig
    g_paramx:copy(paramx_orig)
    local stat = train_batch()
    return g_paramdx, stat
end

-- EVERYTHING ABOVE RUNS ON THREADS

function train(N)
    for n = 1, N do
        local ep = #g_log + 1
        if g_opts.curriculum_end > 0 then
            -- adjust curriculum
            assert(g_opts.curriculum_end > g_opts.curriculum_sta)
            local h = (ep - g_opts.curriculum_sta)/(g_opts.curriculum_end - g_opts.curriculum_sta)
            h = math.min(1, math.max(0, h))
            g_factory:set_hardness(h)
            if g_opts.nworker > 1 then
                for w = 1, g_opts.nworker do
                    g_workers:addjob(w,
                        function(hh)
                            g_factory:set_hardness(hh)
                        end,
                        function() end, h)
                end
            end
        end
        local stat = {}
        for k = 1, g_opts.nbatches do
            if g_opts.show then xlua.progress(k, g_opts.nbatches) end
            if g_opts.nworker > 1 then
                g_paramdx:zero()
                for w = 1, g_opts.nworker do
                    g_workers:addjob(w, train_batch_thread,
                        function(paramdx_thread, s)
                            g_paramdx:add(paramdx_thread)
                            merge_stat(stat, s)
                        end, g_opts, g_paramx)
                end
                g_workers:synchronize()
            else
                local s = train_batch()
                merge_stat(stat, s)
            end
            g_update_param(g_paramx, g_paramdx)
        end
        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
            end
        end
        if stat.bl_count ~= nil and stat.bl_count > 0 then
            stat.bl_cost = stat.bl_cost / stat.bl_count
        else
            stat.bl_cost = 0
        end
        stat.step_active = stat.step_active / stat.step_count

        stat.epoch = #g_log + 1
        print(format_stat(stat))

        table.insert(g_log, stat)
        g_save_model()

        if g_opts.plot then
            local p = torch.zeros(#g_log)
            for i = 1, #g_log do
                p[i] = g_log[i].reward
            end
            gnuplot.plot(p)
        end
    end
end

function g_update_param(x, dx)
    dx:div(g_opts.nworker)
    if g_opts.max_grad_norm > 0 then
        if dx:norm() > g_opts.max_grad_norm then
            dx:div(dx:norm() / g_opts.max_grad_norm)
        end
    end
    local f = function(x0) return x, dx end
    if not g_optim_state then g_optim_state = {} end
    local config = {learningRate = g_opts.lrate}
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_state)
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprob_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_state)
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_state)
    else
        error('wrong optim')
    end

    if g_opts.encoder_lut then
        -- zero NIL embedding
        g_modules['encoder_lut'].weight[g_opts.encoder_lut_nil]:zero()
    end
end
