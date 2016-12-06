-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local MazeItem = torch.class('MazeItem')

-- An item can have attributes (optional except type) such as:
--   type: type of the item such as water, block
--   loc: absolute coordinate of the item
--   name: unique name for the item (optional)
-- All attributes are visible to the agents.
function MazeItem:__init(attr)
    self.type = attr.type
    self.name = attr.name
    self.attr = attr
    self.loc = self.attr.loc
    if self.type == 'block' then
        function self:is_reachable() return false end
    elseif self.type == 'door' then
        function self:is_reachable()
            if self.attr.open == 'open' then
                return true
            end
            return false
        end
    else
        function self:is_reachable() return true end
    end
end

function MazeItem:to_sentence(dy, dx, disable_loc)
    local s = {}
    for k,v in pairs(self.attr) do
        if k == 'loc' then
            if not disable_loc then
                local y = self.loc.y - dy
                local x = self.loc.x - dx
                table.insert(s, 'y' .. y .. 'x' .. x)
            end
            if self.abs_loc_visible then
                table.insert(s, 'ay' .. self.loc.y .. 'x' .. self.loc.x)
            end
        elseif type(k) == 'string' and k:sub(1,1) == '_' then
            -- skip
        else
            table.insert(s, v)
        end
    end
    return s
end
