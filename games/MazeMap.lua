-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local colors_ok, colors = pcall(require, 'ansicolors')
local MazeMap = torch.class('MazeMap')

function MazeMap:__init(opts)
    self.height = opts.map_height or 10
    self.width = opts.map_width or 10

    -- Items by x,y location
    self.items = {}
    for y = 1, self.height do
        self.items[y] = {}
        for x = 1, self.width do
            self.items[y][x] = {}
        end
    end

    self.visibility_mask = torch.Tensor(self.height, self.width)
    self.visibility_mask:fill(1)
end

function MazeMap:add_item(item)
    table.insert(self.items[item.loc.y][item.loc.x], item)
end

function MazeMap:remove_item(item)
    local l = self.items[item.loc.y][item.loc.x]
    for i = 1, #l do
        if l[i].id == item.id then
            table.remove(l, i)
            break
        end
    end
end

function MazeMap:get_empty_loc(fat)
    local fat = fat or 0
    local x, y
    for i = 1, 100 do
        y = torch.random(1+fat, self.height-fat)
        x = torch.random(1+fat, self.width-fat)
        local empty = true
        for j, e in pairs(self.items[y][x]) do
            if not e.attr._immaterial then
                empty = false
            end
        end
        if empty then return y, x end
    end
    error('failed 100 times to find empty location')
end

function MazeMap:is_loc_reachable(y, x)
     if y < 1 or x < 1 then
        return false
    elseif y > self.height or x > self.width then
        return false
    end
    local l = self.items[y][x]
    local is_reachable = true
    for i = 1, #l do
        is_reachable = is_reachable and l[i]:is_reachable()
    end
    return is_reachable
end

function MazeMap:is_loc_visible(y, x)
    if self.visibility_mask[y][x] == 1 then
        return true
    else
        return false
    end
end

function MazeMap:print_ascii()
    for y = 0, self.height + 1 do
        local line = '|'
        for x = 1, self.width do
            local s = '  '
            if y == 0 or y == self.height + 1 then
                s = '--'
            else
                for i = 1, #self.items[y][x] do
                    local item = self.items[y][x][i]                    
                    if item.attr._ascii then
                        s = item.attr._ascii:sub(1,2)
                    elseif item.type == 'block' then
                        s = '[]'
                    elseif item.type == 'goal' then
                        s = '$$'
                    elseif item.type == 'agent' then
                        s = '@@'
                    elseif item.type == 'marine' then
                        s = 'MA'
                    elseif item.type == 'medic' then
                        s = 'ME'
                    else
                        s = '??'
                    end
                    if colors_ok then
                        if item.attr._invisible then
                            s = colors('%{dim}' .. s)
                        elseif item.attr._ascii_color then
                            s = colors('%{' .. table.concat(item.attr._ascii_color, ' ') .. '}' .. s)
                        elseif item.type == 'block' then
                            s = colors('%{red}' .. s)
                        elseif item.type == 'agent' then
                            s = colors('%{blue}' .. s)
                        elseif item.type == 'goal' then
                            s = colors('%{yellow}' .. s)
                        end
                    end
                end
            end
            line = line .. s
        end
        line = line .. '|'
        print(line)
    end
end
