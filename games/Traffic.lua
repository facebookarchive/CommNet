-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local Traffic, parent = torch.class('Traffic', 'MazeBase')

function Traffic:__init(opts, vocab)
    parent.__init(self, opts, vocab)

    self.add_rate = opts.add_rate
    self.add_block = opts.add_block
    self.max_agents = opts.max_agents
    self.costs.collision = self.costs.collision
    self.action_delay = opts.action_delay or 0

    self.source_locs = {}
    self.dest_locs = {}
    self.routes = {}
    self:build_roads()

    self.agents = {}
    self.agents_inactive = {}
    self.agents_active = {}
    for i = 1, self.nagents do
        local agent = self:place_item({type = 'agent', 
            name = 'agent' .. i, _ascii = '@' .. i, _ind = i}, 1, 1)        
        agent.attr._invisible = true
        local colors = {'red', 'green', 'yellow', 'blue', 'magenta', 'cyan'}
        agent.attr._ascii_color = { colors[torch.random(#colors)] }
        agent.abs_loc_visible = true
        agent.active = false
        agent.act = function(self, action_id)
            assert(self.active)
            MazeAgent.act(self, action_id)
        end

        agent.action_names = {}  -- id -> name
        agent.action_ids = {}    -- name -> id
        agent.actions = {}       -- id -> func
        agent.nactions = 0
        agent:add_action('gas',
            function(self)
                assert(self.route)
                self.route_pos = self.route_pos + 1
                local y = self.route[self.route_pos].y
                local x = self.route[self.route_pos].x
                self.map:remove_item(self)
                assert(math.abs(self.loc.y - y) + math.abs(self.loc.x - x) == 1)
                self.loc.y = y
                self.loc.x = x
                self.map:add_item(self)
                self.attr._ascii_color[2] = nil
            end)
        agent:add_action('brake',
            function(self)
                self.attr._ascii_color[2] = 'underline'
            end)

        if self.action_delay > 0 then
            agent.act = function(self, action_id)
                if self.action_buffer then
                    local f = self.actions[self.action_buffer]
                    f(self)
                end
                self.action_buffer = action_id
                self.attr.action_buffer = self.action_names[action_id]
            end
        end

        self.agents[i] = agent
        self.agents_inactive[i] = agent
    end
    self.agent = self.agents[1]
    self.ncollision_total = 0
end

function Traffic:add_agent()
    for _, src in pairs(self.source_locs) do
        if #self.agents_active >= self.max_agents then
            return
        end
        local ri = src.routes[torch.random(#src.routes)]
        local route = self.routes[ri]
        if torch.uniform() < self.add_rate then
            if #self.agents_inactive == 0 then
                return
            end
            local r = torch.random(#self.agents_inactive)
            local agent = self.agents_inactive[r]
            if self.add_block and #self.map.items[src.y][src.x] > 0 then
                return
            end
            self.map:remove_item(agent)
            agent.loc.y = src.y
            agent.loc.x = src.x
            table.remove(self.agents_inactive, r)
            agent.active = true        
            agent.attr._invisible = false
            agent.t = 0
            agent.route = route
            agent.route_pos = 1
            agent.attr.route = 'route' .. ri
            self.map:add_item(agent)
            table.insert(self.agents_active, agent)        
            -- agent.attr._ascii = agent.attr._ind .. ri
            agent.attr._ascii = '<>'
        end
    end
end

function Traffic:update()
    parent.update(self)

    self.success_pass = 0
    self.ncollision = 0
    for _, agent in pairs(self.agents) do
        agent.success_pass = 0
        agent.ncollision = 0
    end
    local t = {}
    for _, agent in pairs(self.agents_active) do
        agent.t = agent.t + 1

        if #self.map.items[agent.loc.y][agent.loc.x] > 1 then
            agent.attr._ascii0 = agent.attr._ascii0 or agent.attr._ascii
            agent.attr._ascii = 'XX'
            agent.ncollision = agent.ncollision + 0.5
            self.ncollision = self.ncollision + 0.5
            self.ncollision_total = self.ncollision_total + 0.5
        end

        local dst = agent.route[#agent.route]
        if agent.loc.y == dst.y and agent.loc.x == dst.x then
            agent.success_pass = agent.success_pass + 1
            self.success_pass = self.success_pass + 1
            agent.attr._invisible = true
            agent.active = false
            table.insert(self.agents_inactive, agent)
            self.map:remove_item(agent)
            agent.loc.y = 1
            agent.loc.x = 1
            self.map:add_item(agent)
        else
            table.insert(t, agent)
        end
    end
    self.agents_active = t

    self:add_agent()
end

function Traffic:get_reward(is_last)
    local r = 0
    r = r - self.agent.success_pass * self.costs.pass 
    r = r - self.agent.ncollision * self.costs.collision
    r = r - self.agent.t * self.costs.wait
    return r
end

function Traffic:is_active()
    return self.agent.active
end

function Traffic:is_success()
    if self.ncollision_total > 0 then
        return false
    else
        return true
    end
end