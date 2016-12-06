-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local Crossing, parent = torch.class('Crossing', 'Traffic')

function Crossing:__init(opts, vocab)
    assert(opts.map_height % 2 == 0)
    assert(opts.map_height == opts.map_width)
    self.length = opts.map_height / 2 - 1
    self.road_length = opts.road_length or self.length
    parent.__init(self, opts, vocab)
end

function Crossing:build_roads_part(transform)
    for y = 1 + self.length - self.road_length, self.length do
        local yy, xx = transform(y, self.length)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
    end
    for x = 1 + self.length - self.road_length, self.length-1 do
        local yy, xx = transform(self.length, x)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
    end

    local r
    local yy, xx
    yy, xx = transform(1 + self.length - self.road_length, self.length + 2)
    table.insert(self.dest_locs, {y = yy, x = xx})
    yy, xx = transform(1 + self.length - self.road_length, self.length + 1)
    table.insert(self.source_locs, {y = yy, x = xx, no_dest = #self.dest_locs, routes = {}})

    -- go straight
    r = {}
    for i = 1 + self.length - self.road_length, self.map.height - self.length + self.road_length do
        yy, xx = transform(i, self.length + 1)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

    -- right turn
    r = {}
    for i = 1 + self.length - self.road_length, self.length + 1 do
        yy, xx = transform(i, self.length + 1)
        table.insert(r, {y = yy, x = xx})
    end
    for i = self.length, 1 + self.length - self.road_length, -1 do
        yy, xx = transform(self.length + 1, i)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

    -- left turn
    r = {}
    for i = 1 + self.length - self.road_length, self.length + 2 do
        yy, xx = transform(i, self.length + 1)
        table.insert(r, {y = yy, x = xx})
    end
    for i = self.length + 2, self.map.width - self.length + self.road_length do
        yy, xx = transform(self.length + 2, i)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)
end

function Crossing:build_roads()
    -- build Crossing
    self:build_roads_part(function(y, x)
        return y, x
        end)
    self:build_roads_part(function(y, x)
        return x, self.map.height + 1 - y
        end)
    self:build_roads_part(function(y, x)
        return self.map.width + 1 - x, y
        end)
    self:build_roads_part(function(y, x)
        return self.map.height + 1 - y, self.map.width + 1 - x
        end)
end
