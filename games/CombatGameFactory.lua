-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local CombatGameFactory = torch.class('CombatGameFactory')

function CombatGameFactory:__init(opts,vocab)
    self.max_agents = opts.number_agent_names
    self.permute_names = opts.permute_names
    self.glist = {}
    self.iglist = {}
    self.gameopts = {}
    self.game_setups = {}
    self.team_control = opts.team_control
    self.vocab = vocab
    self.display = opts.display
    self.generic_opts = opts.generic
    -- how far each unit is from its teams center at init
    self.dispersion = opts.dispersion or 3
    local count = 0
    for i,j in pairs(opts.game_setups) do
        count = count + 1
        self.glist[count] = i
        self.iglist[i] =count
        self.game_setups[i] = j
    end
    self.agent_factory = CombatAgentFactory(opts.agent_factory)
    self.ngames = count
    self.probs = torch.ones(self.ngames)
    if opts.consistent_actions then
        self.consistent_actions = {}
        for i,j in pairs(opts.consistent_actions) do
            self.consistent_actions[i] = j
        end
    end
end

function CombatGameFactory:init_game(gname)
    local out_opts = {}
    out_opts.display = self.display
    out_opts.agents = {}
    local p = torch.randperm(self.max_agents)
    if not self.permute_names then
        for i = 1, self.max_agents do
            p[i] = i
        end
    end
    local count = 0
    out_opts.setup = {}
    out_opts.setup.agent_names = {}
    out_opts.setup.agent_teams = {}
    out_opts.setup.auto_act = {}
    for team, atypes in pairs(self.game_setups[gname]) do
        for a, c in pairs(atypes) do
            local n = torch.random(c[1],c[2])
            for l = 1, n do
                count = count + 1
                out_opts.agents[count] = {}
                out_opts.agents[count].type = a
                local name = 'agent' .. p[count]
                out_opts.agents[count].name = name
                out_opts.setup.agent_names[count] = name
                out_opts.setup.agent_teams[name] = team
                out_opts.setup.auto_act[name] = false
                if self.team_control[team] == 'auto' then
                    out_opts.setup.auto_act[name] = true
                end
            end
        end
    end
    out_opts.setup.consistent_actions = self.consistent_actions
    for i,j in pairs(self.generic_opts) do
        out_opts[i] = j
    end
    local g = CombatGame(out_opts, self.vocab, self.agent_factory)
    return g
end

function CombatGameFactory:init_random_game()
    local n = torch.multinomial(self.probs,1)[1]
    local gname = self.glist[n]
    local g = self:init_game(gname)
    g.gname = gname
    return g
end
