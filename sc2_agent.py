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
        self.stepper_waiting = False
        self.stepper_started = False
        self.stepped = False
        self.obs = None
        self.actrChunks = []
        #self.actr.add_command("tic",self.do_tic)

    def actr_setup(self,actr):
        self.actr = actr

        self.actr.add_command("tic", self.do_tic)
        self.actr.add_command("ignore", self.ignore)

    def ignore(self):
        return 0

    def do_tic(self):
        #print("do_tic: tic called")
        #print("do_tic: waiting for the stepper to start")
        while not self.stepper_started:
            pass
        #print("do_tic: stepper has started")
        #once it's started, it can stop waiting
        while self.stepper_waiting is None:
            pass
        self.stepper_waiting = False
        #print("do_tic: stepper_waiting set to False")
        #the stepper will then do a commmand (that it ought to receive from ACTR)


        #self.tickable=True
        #print("TICKABLE set to TRUE")

        #wait for it to finish stepping
        #print("do_tic: waiting for stepper to step")
        while not self.stepped:
        #    print ("stepped", self.stepped)
            pass
        self.stepped = False
        #print("do_tic: stepped set to FALSE")

        self.stepper_started = False
        self.stepper_waiting = None

        return 1

    def push_observation(self, args):
        '''Return a dictionary of observations'''
        player_relative = self.obs.observation["screen"][_PLAYER_RELATIVE]
        neutral_x, neutral_y = (player_relative == _PLAYER_NEUTRAL).nonzero()
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()

        if neutral_y.any():
            # print(neutral_y, len(neutral_y), min(neutral_y), max(neutral_y))
            chk = self.actr.define_chunks(['neutral_x', neutral_x.mean(), 'neutral_y', neutral_y.mean()])
            self.actrChunks.append(chk)

            #self.actr.schedule_simple_event_now("ignore")
            #self.actr.set_buffer_chunk('imaginal', chk[0])
            self.actr.schedule_simple_event_now("set-buffer-chunk", ['imaginal', chk[0]])#self.actr.set_buffer_chunk('imaginal',chk[0])
            #self.actr.schedule_simple_event_now("ignore")



                    # r_dict = {"neutral_y":int(neutral_y.mean()),"neutral_x":int(neutral_x.mean()),"enemy_y":int(enemy_y.mean()),"enemy_x":int(enemy_x.mean()),
                    #         "player_y":int(player_y.mean()),"player_x":int(player_x.mean())}
        #return r_dict

    def step(self, obs):
        #print("step: step called")
        self.stepper_started = True
        #print("step: set stepper_started to True")
        self.obs = obs
        self.push_observation(None)


        self.stepper_waiting = True
        #print("step: set stepper_waiting to True")
        #print("step: and step is waiting")
        while self.stepper_waiting:
            pass
        #print("step: step has finished waiting")
        #self.tickable = False
        #self.obs = obs
        #self.push_observation(None)
        #while not self.tickable:
            #print("tickable-b", self.tickable)
        #    pass

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

                self.stepped = True
                #print("step: set STEPPED to TRUE")
                return actions.FunctionCall(_NO_OP, [])
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            self.stepped = True
            #print("step: set STEPPED to TRUE")
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            self.stepped = True
            #print("step: set STEPPED to TRUE")
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])



