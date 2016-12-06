-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local CombatGame, parent = torch.class('CombatGame', 'MazeBase')

function CombatGame:__init(opts, vocab, agent_factory)
    parent.__init(self, opts, vocab)
    if opts.cover_map == 1 then
        self.cover_map = torch.zeros(self.map.height,self.map.width)
        local bks = self.items_bytype['block']
        if bks then
            for i,j in pairs(bks) do
                self.cover_map[j.loc.y][j.loc.x] = 1
            end
        end
    end
    self.max_attributes = opts.max_attributes or 10
    self.costs.draw = opts.costs.draw or .5
    self.costs.loss = opts.costs.loss or 1
    self.costs.enemy_health = opts.costs.enemy_health or 0
    self.water_risk_factor = 0
    self.agent_factory = agent_factory
    -- default fog is 0, meaning every agent can see every other
    -- agent.  fog = 1 means each agent can see any agent
    -- that is within sight of any agent on their team
    -- fog = 2 means each agent can see only things within its sight
    -- sight defaults large, lower in config if desired
    self.fog = opts.fog or 0
    self.distance_measure = opts.distance_measure or 2

    self.initial_dispersion = opts.dispersion
    self:place_agents(opts.agents, opts.setup)
end

function CombatGame:place_agents(roster, setup)
    -- this is destructive, it replaces whatever players were there before
    if self.agents then
        for s =1 , #self.agents do
            self:remove_item(self.agents[s])
        end
    end
    self.agents = {}
    self.agents_byname = {}
    self.teams = {}
    for i,j in pairs(roster) do
        self.agents[i] = self.agent_factory:init_agent(j.type,setup,j.name,self)
        self.agents_byname[j.name] = self.agents[i]
        if not self.teams[self.agents[i].team] then
            self.teams[self.agents[i].team] = {j.name}
        else
            table.insert(self.teams[self.agents[i].team],j.name)
        end
        if self.agents[i].type == 'marine' then
            if self.agents[i].attr._auto_act then
                self.agents[i].attr._ascii = 'm' .. i
            else
                self.agents[i].attr._ascii = 'M' .. i
            end
        end
    end
    local d = self.initial_dispersion
    for i,j in pairs(self.teams) do
        local H = torch.random(self.map.height)
        local W = torch.random(self.map.width)
        for s,t in pairs(j) do
            local h = torch.random(-d,d) + H
            h = math.min(math.max(h,1),self.map.height)
            local w = torch.random(-d,d) + W
            w = math.min(math.max(w,1),self.map.width)
            local a = self.agents_byname[t]
            a.loc.x = w
            a.attr.loc.x = w
            a.loc.y = h
            a.attr.loc.y = h
            self:add_prebuilt_item(a)
        end
    end
    self.nagents = #self.agents
    self.agent = self.agents[1]

    for i = 1, self.nagents do
        self.agents[i].abs_loc_visible = true
    end
end

function CombatGame:update()
    if self.finished then return end
    parent.update(self)
    local initiative = torch.randn(self.nagents)
    for s = 1, self.nagents do
        initiative[s] = initiative[s] + self.agents[s].initiative
    end

    -- low initiative means earlier turn
    local _, turn_order = torch.sort(initiative,1)
    for s = 1, self.nagents do
        self.agents[turn_order[s]]:update()
    end
    -- make sure all status updates are done after actions.
    for s = 1, self.nagents do
        self.agents[turn_order[s]]:update_status()
    end

    local teams_alive = {}
    for _, a in pairs(self.agents) do
        if not a.killed then
            teams_alive[a.team] = true
        end
    end
    local count = 0
    local alive
    for i, j in pairs(teams_alive) do
        if j then
            count = count + 1
        end
        alive = i
    end
    if count <= 1 then self.finished = true end
    if count == 1 then
        for s,t in pairs(self.teams[alive]) do
            self.agents_byname[t].won = true
        end
    end
    if count == 0 then
        for s,t in pairs(self.agents) do
            t.draw = true
        end
    end
end

function CombatGame:get_terminal_reward(agent)
    local agent = agent or self.agent
    local reward = 0

    local enemy_total_hp = 0
    for _, a in pairs(self.agents) do
        if a.auto_act and not a.killed then
            enemy_total_hp = enemy_total_hp + a.health
        end
    end
    reward = reward - enemy_total_hp * self.costs.enemy_health

    if not self.finished or agent.draw then
        reward = reward - self.costs.draw
    elseif not agent.won then
        reward = reward - self.costs.loss
    end

    return reward
end

-- no dependence on agent- it may get healed/resurrected/respawned
function CombatGame:is_active(agent)
    local agent = agent or self.agent
    if self.finished then
        return false
    else
        return true
    end
end

function CombatGame:is_success(agent)
    local agent = agent or  self.agent
    if agent.won then
        return true
    else
        return false
    end
end
