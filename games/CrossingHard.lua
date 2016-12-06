-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local CrossingHard, parent = torch.class('CrossingHard', 'Traffic')

function CrossingHard:__init(opts, vocab)
    assert(opts.map_height == 18)
    assert(opts.map_height == opts.map_width)
    parent.__init(self, opts, vocab)
end

function CrossingHard:build_roads_part(transform)
    local yy, xx
    for y = 1, 4 do
        yy, xx = transform(y, 4)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
    end
    for y = 7, 12 do
        yy, xx = transform(y, 4)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
    end
    for x = 1, 3 do
        yy, xx = transform(4, x)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
        yy, xx = transform(7, x)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
        yy, xx = transform(12, x)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
    end
    for y = 7, 11 do
        yy, xx = transform(y, 7)
        self:place_item({type = 'block', _invisible = true}, yy, xx)
    end


    local r
    yy, xx = transform(5, 1)
    table.insert(self.dest_locs, {y = yy, x = xx})
    yy, xx = transform(6, 1)
    table.insert(self.source_locs, {y = yy, x = xx, no_dest = #self.dest_locs, routes = {}})

    -- 1st left
    r = {}
    for x = 1, 6 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 5, 1, -1 do
        yy, xx = transform(y, 6)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 2nd left
    r = {}
    for x = 1, 14 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 5, 1, -1 do
        yy, xx = transform(y, 14)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

    -- go straight
    r = {}
    for x = 1, 18 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 1st right, then left
    r = {}
    for x = 1, 5 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 7, 14 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 6, 18 do
        yy, xx = transform(14, x)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 2nd right, then straight
    r = {}
    for x = 1, 13 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 7, 18 do
        yy, xx = transform(y, 13)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 1st right, then straight
    r = {}
    for x = 1, 5 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 7, 18 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 1st right, then right
    r = {}
    for x = 1, 5 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 7, 13 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 4, 1, -1 do
        yy, xx = transform(13, x)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)


    yy, xx = transform(1, 6)
    table.insert(self.dest_locs, {y = yy, x = xx})
    yy, xx = transform(1, 5)
    table.insert(self.source_locs, {y = yy, x = xx, no_dest = #self.dest_locs, routes = {}})

    -- 1st right
    r = {}
    for y = 1, 5 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 4, 1, -1 do
        yy, xx = transform(5, x)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 2nd right
    r = {}
    for y = 1, 13 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 4, 1, -1 do
        yy, xx = transform(13, x)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

    -- go straight
    r = {}
    for y = 1, 18 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 1st left, then right
    r = {}
    for y = 1, 6 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 6, 13 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 7, 18 do
        yy, xx = transform(y, 13)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 2nd left, then straight
    r = {}
    for y = 1, 14 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 6, 18 do
        yy, xx = transform(14, x)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 1st left, then straight
    r = {}
    for y = 1, 6 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 6, 18 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

   -- 1st left, then left
    r = {}
    for y = 1, 6 do
        yy, xx = transform(y, 5)
        table.insert(r, {y = yy, x = xx})
    end
    for x = 6, 14 do
        yy, xx = transform(6, x)
        table.insert(r, {y = yy, x = xx})
    end
    for y = 5, 1, -1 do
        yy, xx = transform(y, 14)
        table.insert(r, {y = yy, x = xx})
    end
    table.insert(self.routes, r)
    table.insert(self.source_locs[#self.source_locs].routes, #self.routes)

end

function CrossingHard:build_roads()
    -- build 
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
