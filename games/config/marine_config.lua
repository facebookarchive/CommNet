-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--format:
--(current_min,current_max,min_max,max_max,increment,is_continuous)
--not yet using min_max and max_max, not attached to curriculum yet
--is_continuous is 0 if the parameter is an integer
--cooldowns/4 from sc
local c = {}
c._pmiss = torch.Tensor{0,0,0,0,0,1}
c._range = torch.Tensor{1,1,4,4,0,0}
c._attack_type = 'nearest'
c._cooldown_max = torch.Tensor{1,1,4,4,0,0}
c._attack_power = torch.Tensor{1,1,6,6,0,0}
c._health = torch.Tensor{3,3,40,40,0,0}
c._max_health = torch.Tensor{3,3,40,40,0,0}
c._move_speed = torch.Tensor{1,1,1,1,0,0}
c._armor = torch.Tensor{0,0,0,0,0,1}
c._spell_resist = torch.Tensor{0,0,0,0,0,1}
c.type = 'marine'
c._max_shields = torch.Tensor{0,0,0,0,0,1}
c._shields = torch.Tensor{0,0,0,0,0,1}
c._sight = torch.Tensor{g_opts.visibility,g_opts.visibility,4,4,0,0}
return c
