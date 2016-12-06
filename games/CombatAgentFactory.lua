-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local CombatAgentFactory = torch.class('CombatAgentFactory')
if not CombatAgent then paths.dofile('CombatAgent.lua') end

local su = paths.dofile('shooting_utils.lua')

function CombatAgentFactory:__init(opts)
    -- the configs contains both custom functions and
    -- basic setup information.
    -- each config corresponds to an agent type.
    self.configs = {}
    for i,j in pairs(opts.configs) do
        self.configs[i] = j
    end
    self.list = {}
    self.ilist = {}
    self.generate_attr = {}
    local count = 0
    for i,j in pairs(self.configs) do
        count = count + 1
        self.list[count] = i
        self.ilist[i] = count
        local c = paths.dofile(j.config_path)
        local cf = {}
        if j.custom_functions_path then
            cf = paths.dofile(j.custom_functions_path)
        end
        --TODO allow loading custom generate_attr from config
        local generate_attr = function(game_setup, name)
            local attr = {}
            attr.loc = {}
            attr.loc.x = 1
            attr.loc.y = 1
            attr._su = su
            -- the following things come from the game building the agent
            -- and may need to be consistent across several agents
            -- they are not in the agent config.
            attr._agent_names = {}
            for s,t in pairs(game_setup.agent_names) do
                attr._agent_names[s] = t
            end
            attr._agent_teams = {}
            for s,t in pairs(game_setup.agent_teams) do
                attr._agent_teams[s] = t
            end
            attr._auto_act = game_setup.auto_act[name]
            attr.name = name
            attr._consistent_actions = game_setup.consistent_actions

            -- the following things come from the agent config.
            if j.custom_functions_path then
                attr._custom_actions = {}
                attr._custom_add_actions_targeted = {}
                attr._add_shoot_action = cf.add_shoot_action
                attr._choose_action = cf.choose_action
                attr._respawn_function = cf.respawn_function
                attr._regen_function = cf.regen_function
                attr._killed_function = cf.killed_function
                for s,t in pairs(cf.custom_actions) do
                    attr._custom_actions[s] = t
                end
                for s,t in pairs(cf.custom_add_actions_targeted) do
                    attr._custom_add_actions_targeted[s] = t
                end
            end
            --flat config...
            -- config format:
            -- each entry is either a tensor of length 6
            --(current_min,current_max,min_max,max_max,increment,is_continuous)
            --not yet using min_max and max_max, not attached to curriculum yet
            --is_continuous is 0 if the parameter is an integer
            -- or
            -- a string.
            for i,j in pairs(c) do
                if type(j) == 'string' then
                    attr[i] = j
                else
                    if j[6] == 1 then
                        local r = j[2]-j[1]
                        attr[i] = torch.uniform()*r + j[1]
                    else
                        attr[i] = torch.random(j[1],j[2])
                    end
                end
            end
            return attr
        end
        self.generate_attr[i] = generate_attr
    end
end

function CombatAgentFactory:init_agent(atype,game_setup,name,maze)
    -- agent_type is in 1,...,#self.list
    -- and corresponds ot a particular config
    local attr = self.generate_attr[atype](game_setup,name)
    local a = CombatAgent(attr,maze)
    return a
end

function CombatAgentFactory:init_random_agent(game_setup,name,maze)
    local n = torch.multinomial(self.probs,1)[1]
    local agent_type = self.list[n]
    local attr = self.generate_attr[agent_type](game_setup,name)
    local a = CombatAgent(attr,maze)
    return a
end
