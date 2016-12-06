-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

paths.dofile('MazeBase.lua')
paths.dofile('OptsHelper.lua')
paths.dofile('GameFactory.lua')
paths.dofile('batch.lua')

-- for traffic games
paths.dofile('Traffic.lua')
paths.dofile('Crossing.lua')
paths.dofile('CrossingEasy.lua')
paths.dofile('CrossingHard.lua')

-- for combat games
paths.dofile('CombatAgent.lua')
paths.dofile('CombatAgentFactory.lua')
paths.dofile('CombatGame.lua')
paths.dofile('CombatGameFactory.lua')


local function init_game_opts()
    if g_opts.team_control then
        -- combat game
        g_factory = CombatGameFactory(g_opts,g_vocab)
        return
    end

    local games = {}
    local helpers = {}
    games.Crossing = Crossing
    helpers.Crossing = OptsHelper
    games.CrossingEasy = CrossingEasy
    helpers.CrossingEasy = OptsHelper
    games.CrossingHard = CrossingHard
    helpers.CrossingHard = OptsHelper
    games.CombatGame = CombatGame
    helpers.CombatGame = CombatGameFactory

    g_factory = GameFactory(g_opts,g_vocab,games,helpers)

    return games, helpers
end

function g_init_vocab()
    local function vocab_add(word)
        if g_vocab[word] == nil then
            local ind = g_opts.nwords + 1
            g_opts.nwords = g_opts.nwords + 1
            g_vocab[word] = ind
            g_ivocab[ind] = word
        end
    end
    g_vocab = {}
    g_ivocab = {}
    g_ivocabx = {}
    g_ivocaby = {}
    g_opts.nwords = 0

    -- general
    vocab_add('nil')
    vocab_add('agent')
    for i = 1, 5 do
        vocab_add('agent' .. i)
    end
    vocab_add('goal')
    if g_opts.nactions_comm > 1 then
        for i = 1, g_opts.nactions_comm do
            vocab_add('talk' .. i)
        end
    end

    -- absolute coordinates
    for y = 1, 10 do
        for x = 1, 10 do
            vocab_add('ay' .. y .. 'x' .. x)
        end
    end

    -- game specific
    if g_factory.games then
        if g_factory.games['Crossing'] or g_factory.games['CrossingEasy']
            or g_factory.games['CrossingHard'] then
            vocab_add('block')
            for i = 1, 12 do
                vocab_add('route' .. i)
            end
            for y = 1, 14 do
                for x = 1, 14 do
                    vocab_add('ay' .. y .. 'x' .. x)
                end
            end
            for i = 1, 10 do
                vocab_add('agent' .. i)
            end
            vocab_add('gas')
            vocab_add('brake')
            if g_factory.games['CrossingHard'] then
                for y = 1, 18 do
                    for x = 1, 18 do
                        vocab_add('ay' .. y .. 'x' .. x)
                    end
                end
                for i = 1, 7*8 do
                    vocab_add('route' .. i)
                end
                for i = 1, 20 do
                    vocab_add('agent' .. i)
                end
            end
        end
    else
        -- for combat games
        for i = 1, 10 do
            vocab_add('agent' .. i)
        end
        vocab_add('marine')
        for s = -3, 10 do
           vocab_add('health' .. s)
        end
        for s = 0, 2 do
           vocab_add('shields' .. s)
        end
        for s = 0, 2 do
           vocab_add('mana' .. s)
        end
        for s = 0, 5 do
           vocab_add('cooldown' .. s)
        end
        for s = 1, 5 do
            vocab_add('team' .. s)
        end

        for y = 1, 15 do
            for x = 1, 15 do
                vocab_add('ay' .. y .. 'x' .. x)
            end
        end
    end
end

function g_init_game()
    g_opts = dofile(g_opts.games_config_path)
    local games, helpers = init_game_opts()
end

function new_game()
    if g_opts.game == nil or g_opts.game == '' then
        return g_factory:init_random_game()
    else
       return g_factory:init_game(g_opts.game)
    end
end
