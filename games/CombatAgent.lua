-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local CombatAgent, parent = torch.class('CombatAgent','MazeItem')
-- extensible agent for shooting games.
--
-- default fields:
-- team (used for controlling friendly fire/enemy healing etc...
--     should be one of 'team1', 'team2' ... )
-- range (shot range, l2 or l0 depending on distance_measure=2 or 0)
-- attack_power (number of hp or shields to take away from enemy on hit)
-- health
-- sight; used by other agents in :is_visible_to .  sight defaults to 9999
-- cooldown (for shots; custom spell cooldowns should be added separately)
-- spell_resist (a number from 0 to 1 giving a percent reduction in health removed
--     from shots. Spells may also take it into account)
-- armor (a number from 0 to 1 giving a percent reduction in health removed
--     by shots)
-- shields (same as health, but regenerates.  by default damage is taken from
--     shields first)
-- shield_regen_rate (rate of shield regen.  1 is fastest)
-- shield_regen (amount of shield regen.  how much to add)
-- mana (not used in default agent)
-- mana_regen_rate (not used in default agent, as in shield_regen_rate)
-- mana_regen (not used in default agent, acts in analogy with shield_regen)
-- hit_type describes what kind of shots cause the agent damage, and
-- shot_type describes what kind of shots the agent uses.  these
--   should be a table with the type as key and
--   a scalar multiplier as value.  empty entries correspond to 0
--   damage by default is calculated with the max of shot_type*hit_type
--
-- defaults to basic shooting and moving.
-- can add modified shoot, move, and spells via fields in attr
--
-- buffs go in self.buffs; each buff should be a function that takes self as an input.
-- buffs are applied before any other update.
-- buff functions are responsible for writing their status
-- to attr and removing their status from attr.
-- buffs can be used to make other information visible in to_sentence
-- by using them just to write to attr
--
-- This can be used for a model or human controlled agent or for a
-- rule based npc.  use self.auto = true to use rule based behavior
-- and self.auto = false and the :act(action_id) method for model
-- input.  TODO auto, model, or byhand
--
-- attr.loc should be initially set by the game, as the factory will put it at 1,1
--
-- (max) number of agents in the maze is fixed at init,
-- and the (possible) names are explicitly passed in attr._agent_names
-- for now (TODO fix this?) team names should be passed in as well
-- as in if i,j in pairs(attr._agent_names] then
-- attr._team_names[j] gives the team of that agent.
-- default is to not allow friendly fire.
--
-- attr._consistent_actions is a table with a list of
-- every possible action other agents using the same controller
-- might use.  if it doesn't exist, actions are packed into
-- as few slots as necessary; otherwise, actions are put into
-- the slots given by the table (and so many actions will do nothing).
-- This way a single model can control many types of agents.
--
-- CombatAgentFactory loads configs and dofiles for filling fields in attr
-- TODO: clean up attr._ fields and make a standard self.opts that
-- can be read into

--local su = paths.dofile('shooting_utils.lua')

function CombatAgent:__init(attr,maze)
    -- updateable is false because we will update the
    -- agents by hand in the game update, as general
    -- MazeBase doesn't have capacity to deal with initiative
    self.updateable = false
    self.initiative = attr._initiative or 1

    -- sight defaults large, lower in config if desired
    self.sight = attr._sight or 9999

    self.auto_act = attr._auto_act
    self.su = attr._su
    self.killed = false

    self.closest_agent = 0
    self.range_closest_agent = 0

    self.agent_names = attr._agent_names
    self.agent_teams = attr._agent_teams
    self.nagents = #self.agent_names
    self.type = attr.type or 'CombatAgent'
    self.name = attr.name
    self.agents_byname = {}
    for i,j in pairs(self.agent_names) do
        self.agents_byname[j] = i
    end
    assert(self.agents_byname[self.name])
    self.team = self.agent_teams[self.name]

    self.attack_type = attr._attack_type or 'nearest' --nearest, random, follow
    self.action_noise = attr._action_noise or 0 --FIXME! not in factory
    self.ignore_range = attr._ignore_range or 9999 --FIXME! not in factory

    self.shot_type = attr._shot_type or {ground = 1}
    self.hit_type = attr._hit_type or {ground = 1}


    self.pmiss = attr._pmiss or 0
    self.attack_power = attr._attack_power or 1
    self.range = attr._range or 1
    self.health = attr._health
    self.move_speed = attr._move_speed or 1
    assert(self.move_speed <= 1)
    self.armor = attr._armor or 0
    self.spell_resist = attr._spell_resist or 0


    self.maze = maze
    self.cover_map = maze.cover_map
    self.map = maze.map
    self.attr = attr
    self.loc = self.attr.loc
    self.distance_measure = self.maze.distance_measure

    self.nactions = 0
    self.action_names = {}
    self.action_ids ={}
    self.actions ={}
    if attr._consistent_actions then
        for i,j in pairs(attr._consistent_actions) do
            self.action_names[i] = j
            self.action_ids[j] = i
            self.nactions = self.nactions + 1
            self.actions[i] = function() end
        end
    end

    self:add_move_actions()
    self.buffs = {} --put all temporary spell effect functions here.
    self.add_shoot_action = attr._add_shoot_action or self.add_default_shoot_action
    for i,j in pairs(self.agent_names) do
        self.add_shoot_action(self,j)
    end
    self:add_action('fumble', function(self) end)
    -- these are for custom actions that do not take a specific agent or npc target
    if attr._custom_actions then
        for i,j in pairs(attr._custom_actions) do
            self:add_action(i,j)
        end
    end
    -- these are for custom actions that take a target, e.g. healing spell
    if attr._custom_add_actions_targeted then
        for s,t in pairs(attr._custom_add_actions_targeted) do
            for i,j in pairs(self.agent_names) do
                t(self,j)
            end
        end
    end
    self.choose_action = attr._choose_action or self.default_choose_action
    self.respawn_function = attr._respawn_function or self.default_respawn
    self.killed_function = attr._killed_function or self.default_killed
    self.regen_function = attr._regen_function or self.default_regen

    self.shields = {}
    self.shields.counter = 0
    self.shields.shields = attr._shields or attr._max_shields or 0
    self.shields.regen = attr._shields_regen or 0
    self.shields.regen_rate = attr._shields_regen_rate or 1
    self.shields.max_shields = attr._max_shields or 0

    self.mana = {}
    self.mana.counter = 0
    self.mana.mana = attr._mana or attr._max_mana or 0
    self.mana.regen = attr._mana_regen or 0
    self.mana.regen_rate = attr._mana_regen_rate or 1
    self.mana.max_mana = attr._max_mana or 0

    self.cooldown = attr.cooldown or 0
    self.cooldown_max = attr._cooldown_max

    self.attr.health = 'health' .. self.health
    self.attr.mana = 'mana' .. self.mana.mana
    self.attr.cooldown = 'cooldown' .. self.cooldown
    self.attr.shields = 'shields' .. self.shields.shields
    self.attr.team = self.team
    function self:is_reachable() return false end
end


function CombatAgent.default_respawn(self) end

-- regen function is responsible for updating all
-- counters that automatically increment while alive
function CombatAgent.default_regen(self)
    self.cooldown = math.max(self.cooldown - 1, 0)
    local sh = self.shields
    sh.counter = sh.counter + 1
    if sh.max_shields > 0 and sh.counter >= sh.regen_rate then
        sh.shields = math.min(sh.shields,sh.max_shields)
        sh.counter = 0
    end
    local ma = self.mana
    ma.counter = ma.counter + 1
    if ma.max_mana > 0 and ma.counter >= ma.regen_rate then
        ma.mana = math.min(ma.mana,ma.max_mana)
        ma.counter = 0
    end
end


function CombatAgent:default_choose_action()
    if self.action_noise > math.random() then
        -- take random action
        local m = torch.random(#self.action_ids)
        return m
    else
        -- figure out relative position to each agent on other teams
        local rel_total = torch.Tensor(self.maze.nagents)
        local rel_x = torch.Tensor(self.maze.nagents)
        local rel_y = torch.Tensor(self.maze.nagents)
        for i, agent in pairs(self.maze.agents) do
            rel_x[i] = agent.loc.x - self.loc.x
            rel_y[i] = agent.loc.y - self.loc.y
            rel_total[i] = self:distance_to(agent)
            if agent.health <= 0 then
                flag = true
                if self.follow_target and self.follow_target == i then
                    self.follow_target = nil
                end
                rel_total[i] = 99999 -- to prevent enemy homing in on already dead agent
            end
            if self.team == agent.team then
                rel_total[i] = 99999 -- don't shoot teammates
            end
        end
        -- figure out closest one
        local rel_total_min, ind = torch.min(rel_total,1)
        if self.attack_type == 'nearest' then
              self.closest_agent = ind[1]
            self.range_closest_agent = rel_total_min[1]
        elseif self.attack_type == 'follow' then
            if not self.follow_target then
                self.closest_agent = ind[1]
                self.range_closest_agent = rel_total_min[1]
                self.follow_target = ind[1]
            else
                self.closest_agent = self.follow_target
                self.range_closest_agent = rel_total[self.follow_target]
            end
        else
            self.closest_agent = torch.random(self.maze.nagents)
            self.range_closest_agent = rel_total[self.closest_agent]
        end
        local target = self.maze.agents[self.closest_agent]
        local target_name = target.name
        if rel_total_min[1] <= self.range and self.cooldown == 0 then
            return self.action_ids['shoot_' .. target_name]
        else
            if not target:is_visible_to(self) then return self.action_ids['fumble'] end
            -- move towards agent.
            if math.random() > (math.abs(rel_x[self.closest_agent])/self.range_closest_agent) then
                if rel_y[self.closest_agent] > 0 then
                    return self.action_ids['down']
                else
                    return self.action_ids['up']
                end
            else
                if rel_x[self.closest_agent] > 0 then
                    return self.action_ids['right']
                else
                    return self.action_ids['left']
                end
            end
        end
    end
end

function CombatAgent:add_action(name, f)
    if not self.action_ids[name] then
        self.nactions = self.nactions + 1
        self.action_names[self.nactions] = name
        self.actions[self.nactions] = f
        self.action_ids[name] = self.nactions
    else
        self.actions[self.action_ids[name]] = f
    end
end


function CombatAgent:add_move_actions()
    self:add_action('up',
        function(self)
            -- check if the enemy fumbles; low move speed means
            -- high fumble probability:
            if math.random() > self.move_speed then return end
            if self.map:is_loc_reachable(self.loc.y - 1, self.loc.x) then
                self.map:remove_item(self)
                self.loc.y = self.loc.y - 1
                self.map:add_item(self)
            end
        end)
    self:add_action('down',
        function(self)
            if math.random() > self.move_speed then return end
            if self.map:is_loc_reachable(self.loc.y + 1, self.loc.x) then
                self.map:remove_item(self)
                self.loc.y = self.loc.y + 1
                self.map:add_item(self)
            end
        end)
    self:add_action('left',
        function(self)
            if math.random() > self.move_speed then return end
            if self.map:is_loc_reachable(self.loc.y, self.loc.x - 1) then
                self.map:remove_item(self)
                self.loc.x = self.loc.x - 1
                self.map:add_item(self)
            end
        end)
    self:add_action('right',
        function(self)
            if math.random() > self.move_speed then return end
            if self.map:is_loc_reachable(self.loc.y, self.loc.x + 1) then
                self.map:remove_item(self)
                self.loc.x = self.loc.x + 1
                self.map:add_item(self)
            end
        end)
end

function CombatAgent:add_default_shoot_action(j)
    if self.agent_teams[j] ~= self.team then
        local function shoot_agent(self)
            local enemy = self.maze.agents_byname[j]
            if self.cooldown > 0 then return end
            self.cooldown = self.cooldown_max + 1
            local sx =  self.loc.x
            local sy =  self.loc.y
            local ex =  enemy.loc.x
            local ey =  enemy.loc.y
            local dist = self:distance_to(enemy)
            local mp = self.maze.cover_map
            local ha,l = self.su.check_hittable(mp,self.range,sy,sx,ey,ex,dist)
            if ha and math.random() > self.pmiss then
                local m = 0
                for i,j in pairs(self.shot_type) do
                    if enemy.hit_type[i] then
                        m = math.max(m,enemy.hit_type[i]*j)
                    end
                end
                local damage = self.attack_power * m
                local sh = enemy.shields
                if sh and sh.shields and sh.shields > 0 then
                    local reduced_shields = math.max(sh.shields - damage, 0)
                    damage = damage - (sh.shields - reduced_shields)
                    sh.shields = reduced_shields
                end
                if enemy.armor then damage=math.max((1-enemy.armor)*damage,0) end
--                agent.health = math.max(agent.health - damage, 0)
                enemy.health = enemy.health - damage --this can be negative.
            else
                return
            end
            if self.maze.display then
                if not self.maze.map.shots then self.maze.map.shots = {} end
                local shot = {l, self.shot_image}
                table.insert(self.maze.map.shots,shot)
            end
        end
        self:add_action('shoot_' .. j, shoot_agent)
    end
end

function CombatAgent.default_killed(self)
    if self.killed == false then
        function self:is_reachable() return true end
        self.killed = true
        self.attr._invisible = true
        self.respawn_counter = 0
    else
        self.respawn_counter = self.respawn_counter + 1
    end
end


function CombatAgent:update()
    if self.attr.max_health then
        self.health = math.min(self.health, self.attr._max_health)
    end
    for i,j in pairs(self.buffs) do j(self) end
    if self.health <= 0 then
        self:killed_function()
        self:respawn_function()
    end
    local act
    if not self.killed then
        if self.auto_act then
            local a = self:choose_action()
            act = self.actions[a]
        else
            act = self.actions[self.action_buffer]
        end
        act(self)
        self:regen_function()
    end
    -- write some fields to attr so they are visible to write in
    -- self:to_sentence...
    self.attr.health = 'health' .. math.ceil(self.health)
    self.attr.mana = 'mana' .. self.mana.mana
    self.attr.cooldown = 'cooldown' .. self.cooldown
    self.attr.shields = 'shields' .. self.shields.shields
    self.attr.loc.x = self.loc.x
    self.attr.loc.y = self.loc.y
    -- to make other fields visible override this function or
    -- use buff functions.
end


function CombatAgent:update_status()
    if self.attr.max_health then
        self.health = math.min(self.health, self.attr._max_health)
    end
    if not self.killed and self.health <= 0 then
        self:killed_function()
    end
    -- write some fields to attr so they are visible to write in
    -- self:to_sentence...
    self.attr.health = 'health' .. math.ceil(self.health)
    self.attr.mana = 'mana' .. self.mana.mana
    self.attr.cooldown = 'cooldown' .. self.cooldown
    self.attr.shields = 'shields' .. self.shields.shields
    self.attr.loc.x = self.loc.x
    self.attr.loc.y = self.loc.y
    -- to make other fields visible override this function or
    -- use buff functions.
end

function CombatAgent:act(action_id)
    -- put action in the action buffer, will actually take
    -- effect after buffs are run
    self.action_buffer = action_id
end


function CombatAgent:is_visible_to(agent)
    -- returns true if agent can see self, false otherwise
    if self.attr._invisible then return false end
    if not self.maze.fog or self.maze.fog == 0 then
        return true
    end
    if self.maze.fog == 2 then
        local vx = agent.loc.x - self.loc.x
        local vy = agent.loc.y - self.loc.y
        if agent:distance_to(self) <= agent.sight then
            return true
        else
            return false
        end
    end
    if self.maze.fog == 1 then
        local visible = false
        for i,a in pairs(self.maze.agents) do
            if a.team == agent.team then
                if a:distance_to(self) <= a.sight then
                    visible = true
                end
            end
        end
        return visible
    end
end


function CombatAgent:to_sentence(dy,dx,disable_loc)
    if self:is_visible_to(self.maze.agent) then
        return parent.to_sentence(self,dy,dx,disable_loc)
    else
        --this wastes a memory slot.  fixme?
        return {}
    end
end

function CombatAgent:distance_to(agent)
    local vx = agent.loc.x - self.loc.x
    local vy = agent.loc.y - self.loc.y
    if self.distance_measure == 0 then
        return math.max(math.abs(vx), math.abs(vy))
    else
        return math.sqrt(vx^2+vy^2)
    end
end
