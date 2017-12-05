# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from twisted.internet import protocol, reactor
from twisted.protocols.basic import LineReceiver
import json
import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class MoveToBeacon(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""

    def __init__(self):
        super().__init__()
        self.tickable = False
        self.obs = None

        #self.actr.add_command("tic",self.do_tic)

    def actr_setup(self,actr):
        self.actr = actr

        self.actr.add_command("tic", self.do_tic)

    def do_tic(self):
        self.tickable=True
        #print("TICKABLE set to TRUE")

    def get_observation(self, args):
        '''Return a dictionary of observations'''
        player_relative = self.obs.observation["screen"][_PLAYER_RELATIVE]
        neutral_x, neutral_y = (player_relative == _PLAYER_NEUTRAL).nonzero()
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()

        if neutral_y.any():
            # print(neutral_y, len(neutral_y), min(neutral_y), max(neutral_y))
            r_dict = {}
            somekeys = ["neutral_y", "neutral_x", "enemy_y", "enemy_x", "player_y", "player_x"]
            for key in somekeys:
                try:
                    r_dict[key] = eval("int(" + key + ".mean())")
                except ValueError:
                    r_dict[key] = "NaN"


                    # r_dict = {"neutral_y":int(neutral_y.mean()),"neutral_x":int(neutral_x.mean()),"enemy_y":int(enemy_y.mean()),"enemy_x":int(enemy_x.mean()),
                    #         "player_y":int(player_y.mean()),"player_x":int(player_x.mean())}
        return r_dict

    def step(self, obs):
        self.tickable = False
        self.obs = obs
        while not self.tickable:
            # print("tickable", self.tickable)
            pass

        # self.tick = False
        # stopwatch.sw.clear()
        # stopwatch.sw.enabled = True
        super(MoveToBeacon, self).step(obs)
        # put the ACT-R loop here
        # while not self.tick:
        #    print("nothing...")
        # time.sleep(2)

        #return actions.FunctionCall(_NO_OP, [])
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not neutral_y.any():

                return actions.FunctionCall(_NO_OP, [])
            target = [int(neutral_x.mean()), int(neutral_y.mean())]

            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:

            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


