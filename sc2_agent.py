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
        self.response = ["_NO_OP", "[]"]
        print("SELECT:", _SELECT_ARMY)
        #self.actr.add_command("tic",self.do_tic)

    def actr_setup(self,actr):
        self.actr = actr

        self.actr.add_command("tic", self.do_tic)
        self.actr.add_command("ignore", self.ignore)
        self.actr.add_command("set_response", self.set_response)
        self.actr.add_command("RHSWait", self.RHSWait)

    def set_response(self,*args):
        print(args)
        args = list(args)

        #Also part of the rest because the production has already been chosen.
        #This will, of course, add some lag because it won't know to click on the agent
        #Until after 1 cycle.
        # if not _MOVE_SCREEN in self.obs.observation["available_actions"]:
        #     self.response = [_NO_OP, []]
        #     self.do_tic()
        #     return 1

        if args[0] == "_SELECT_ARMY":
            self.response = [_SELECT_ARMY, [_SELECT_ALL]]
        elif args[0] == "_MOVE_SCREEN":
            self.response = [_MOVE_SCREEN, [_NOT_QUEUED, [args[2][1],args[3][1]]]]
        else:
            pass

        # self.response = []
        # if len(args) <= 2:
        #     self.response.append(eval(args[0]))
        #     self.response.append(eval(args[1]))
        # else:
        #     self.response.append(eval(args[0]))
        #     args.remove(args[0])
        #     argument2 = []
        #     for x in args:
        #         if isinstance(x,list):
        #             if isinstance(x, str):
        #                 argument2.append(eval(x))
        #             else:
        #                 argument2.append(x[1])
        #         else:
        #             if isinstance(x, str):
        #                 argument2.append(eval(x))
        #             else:
        #                 argument2.append(x)
        #     self.response.append(argument2)
        print("RES:", self.response)
            # argument2 = [x for x in args]
            # self.response.append(argument2)


        #print("set_response: set_response called with response:", response, "arg:", arg)
        #self.response = [response,arg]
        self.do_tic()

        return 1

    def RHSWait(self):
        self.RHSWaitFlag = True
        while self.RHSWaitFlag:
            time.sleep(0.00001)
        return 1

    def ignore(self):
        return 0

    def do_tic(self):
        print("do_tic: tic called")
        #print("do_tic: waiting for the stepper to start")
        self.actr.schedule_simple_event_now("ignore")
        self.tickable = True

        return 1

    def push_observation(self, args):
        '''Return a dictionary of observations'''
        player_relative = self.obs.observation["screen"][_PLAYER_RELATIVE]
        neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()

        if neutral_y.any():
            # print(neutral_y, len(neutral_y), min(neutral_y), max(neutral_y))


            chk = self.actr.define_chunks(['neutral_x', int(neutral_x.mean()), 'neutral_y', int(neutral_y.mean()),'wait', 'false'])
            # the wait, false is for to make sure something other than the wait production fires.
            self.actrChunks.append(chk)

            #self.actr.schedule_simple_event_now("ignore")
            #self.actr.set_buffer_chunk('imaginal', chk[0])
            self.actr.schedule_simple_event_now("set-buffer-chunk", ['imaginal', chk[0]])#self.actr.set_buffer_chunk('imaginal',chk[0])
            #self.actr.schedule_simple_event_now("ignore")

        return 1

                    # r_dict = {"neutral_y":int(neutral_y.mean()),"neutral_x":int(neutral_x.mean()),"enemy_y":int(enemy_y.mean()),"enemy_x":int(enemy_x.mean()),
                    #         "player_y":int(player_y.mean()),"player_x":int(player_x.mean())}
        #return r_dict

    def step(self, obs):
        print("step: step called")
        self.response = [_NO_OP, []]

        #this is a temporary solution for resetting...
        if not _MOVE_SCREEN in obs.observation["available_actions"]:
            current_goal_chunk = self.actr.buffer_chunk('goal')
            self.actr.mod_chunk(current_goal_chunk[0], "state", "select-army")

        #self.stepper_started = True
        #print("step: set stepper_started to True")
        self.obs = obs
        w = self.push_observation(None)
        current_imaginal_chunk = self.actr.buffer_chunk('imaginal')
        #print(current_imaginal_chunk)
        self.actr.mod_chunk(current_imaginal_chunk[0], "wait", "false")
        self.RHSWaitFlag = False
        #self.actr.schedule_simple_event_now("mod-chunk-fct", 'imaginal', 'wait', 'false')
        while not self.tickable:
            time.sleep(0.00001)
            #pass
        #print("step: about to", self.response)
        #return actions.FunctionCall(eval(self.response[0]),eval(self.response[1]))
        argone = self.response[0]#eval(self.response[0])
        argtwo = self.response[1]#eval(self.response[1])

        #I don't think this if is needed anymore.
        # if argone == _NO_OP:
        #     self.stepped = True
        #     self.tickable = False
        #     return actions.FunctionCall(argone, [_SELECT_ALL])


        self.stepped = True
        self.tickable = False
        return actions.FunctionCall(argone,argtwo)

        # if _MOVE_SCREEN in obs.observation["available_actions"]:
        #     player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        #     neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        #     if not neutral_y.any():
        #
        #         self.stepped = True
        #         self.tickable = False
        #         #print("step: set STEPPED to TRUE")
        #         return actions.FunctionCall(_NO_OP, [])
        #     target = [int(neutral_x.mean()), int(neutral_y.mean())]
        #     self.stepped = True
        #     self.tickable = False
        #     #print("step: set STEPPED to TRUE")
        #     return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        # else:
        #     self.stepped = True
        #     self.tickable = False
        #     #print("step: set STEPPED to TRUE")
        #     return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])



