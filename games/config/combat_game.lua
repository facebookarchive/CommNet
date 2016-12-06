-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

g_opts = g_opts or {}
g_opts.number_agent_names = g_opts.nagents * 2
-- currently teams are all auto or all model controlled
-- TODO change this to 'auto', 'model', 'byhand'
g_opts.team_control = {};
g_opts.team_control['team1'] = 'model'
g_opts.team_control['team2'] = 'auto'
g_opts.agent_factory ={}
g_opts.agent_factory.configs = {}

g_opts.generic = {}
g_opts.generic.costs ={}
g_opts.generic.costs.block = 1000
g_opts.generic.costs.step = 0
g_opts.generic.costs.draw = 1
g_opts.generic.costs.loss = 1
g_opts.generic.costs.enemy_health = 0.1
--------
g_opts.generic.map_height = 15
g_opts.generic.map_width = 15
g_opts.generic.dispersion = 5
g_opts.generic.permute_names = false
g_opts.generic.distance_measure = 0
g_opts.generic.fog = 1
g_opts.generic.visibility = g_opts.visibility

---- agent configs:
---- these point to the detailed config file.


-- marine
g_opts.agent_factory.configs['marine'] = {}
g_opts.agent_factory.configs['marine'].config_path = 'config/marine_config.lua'

------------------------------------------------------------------------------------------------------------------------
--++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
------------------------------------------------------------------------------------------------------------------------
-- game setups

-- warning: nothing forces the consistent action names
-- to match up with what you call them in the agent config
g_opts.consistent_actions = {}
g_opts.consistent_actions[1] = 'up'
g_opts.consistent_actions[2] = 'down'
g_opts.consistent_actions[3] = 'left'
g_opts.consistent_actions[4] = 'right'
g_opts.consistent_actions[5] = 'fumble'
for i = 1, g_opts.number_agent_names do
    g_opts.consistent_actions[5+i] = 'shoot_agent' .. i
end

g_opts.nactions = #g_opts.consistent_actions

g_opts.game_setups = {}

-- M vs M
g_opts.game_setups['MM_v_MM'] = {}
s = g_opts.game_setups['MM_v_MM']
s['team1'] = {}
s['team1']['marine'] = torch.Tensor{g_opts.nagents, g_opts.nagents,1,1,0}
s['team2'] = {}
s['team2']['marine'] = torch.Tensor{g_opts.nagents, g_opts.nagents,1,1,0}


return g_opts