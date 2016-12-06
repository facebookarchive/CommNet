-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

if not g_opts then g_opts = {} end

g_opts.multigames = {}

local mapH = torch.Tensor{7,7,7,7,1}
local mapW = torch.Tensor{7,7,7,7,1}

-------------------
--some shared StaticOpts
local sso = {}
-------------- costs:
sso.costs = {}
sso.costs.goal = 0
sso.costs.step = 0
sso.costs.pass = 0
sso.costs.collision = 10
sso.costs.wait = 0.01
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.add_block = false
sso.visibility = g_opts.visibility
sso.nagents = g_opts.nagents

-------------------------------------------------------
local CrossingEasyRangeOpts = {}
CrossingEasyRangeOpts.mapH = mapH:clone()
CrossingEasyRangeOpts.mapW = mapW:clone()
CrossingEasyRangeOpts.add_rate = torch.Tensor{0.1,0.1,0.1,0.3,0.01}
CrossingEasyRangeOpts.max_agents = torch.Tensor{3,3,3,g_opts.nagents,1}

local CrossingEasyStaticOpts = {}
for i,j in pairs(sso) do CrossingEasyStaticOpts[i] = j end

local CrossingEasyOpts ={}
CrossingEasyOpts.RangeOpts = CrossingEasyRangeOpts
CrossingEasyOpts.StaticOpts = CrossingEasyStaticOpts

g_opts.multigames.CrossingEasy = CrossingEasyOpts

return g_opts
