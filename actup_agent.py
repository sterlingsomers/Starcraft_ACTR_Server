from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from pysc2.lib import features
from pysc2.lib import actions
from absl import flags
import numpy as np
import random
import pyactup
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

class ActupAgent():

    def __init__(self,env,noise=0.5,decay=0.5,temperature=1.0,threshold=-100.0,mismatch=1.0,optimized_learning=False):
        self.memory = pyactup.Memory(noise,decay,temperature,threshold,mismatch)
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.env = env
        self.select_army = np.asarray([7],dtype=int)
        self.patrol_screen = np.asarray([333],dtype=int)
        self.episode_counter = 0
        self.action_map = {-1:self.select_green,0:self.select_orange}
        self.player_selected = False
        #actions
        #-1 green
        #0  orange
        #1  around
        # self.memory.learn(green=1,orange=1,action=-1,value=1000)
        # self.memory.learn(green=1,orange=1,action=0,value=1000)
        self.memory.learn(green=0,orange=1,action=0,value=100)
        self.memory.learn(green=1,orange=0,action=-1,value=100)



    def _input_to_feedict(self):
        pass

    def create_obs_dict(self, obs, target=None):
        '''Return a dictionary of observations'''
        #args are:
        #[action_id,spatial_action_2d,value_estimate,fc1_narray]

        player_relative = obs["player_relative_screen"]
        #print("PR", (player_relative == _PLAYER_NEUTRAL).nonzero())
        orange_beacon = 0
        green_beacon = 0
        player = 0
        between = 0
        obs_dict = {}

        neutral_x, neutral_y = (player_relative == _PLAYER_NEUTRAL).nonzero()[1:3]
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()[1:3]
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()[1:3]

        if neutral_y.any():
            green_beacon = 1

            if enemy_y.any():
                orange_beacon = 1
            if player_y.any():
                player = 1

        if enemy_y.any():
            orange_beacon = 1

            if neutral_y.any():
                green_beacon = 1
            if player_y.any():
                player = 1

        if target is not None:

            if orange_beacon and player and green_beacon:
                #check for blocking or overlap

                #Determine if green is between orange and player
                green_points = list(zip(neutral_x,neutral_y))
                orange_points = list(zip(enemy_x, enemy_y))
                player_points = list(zip(player_x, player_y))

                if target == 'green':
                    start = player_points
                    destination = green_points
                    other_points = orange_points
                elif target == 'orange':
                    start = player_points
                    destination = orange_points
                    other_points = green_points
                else:
                    raise ValueError('target is either green or orange')

                #https: // stackoverflow.com / questions / 328107 / how - can - you - determine - a - point - is -between - two - other - points - on - a - line - segment
                between = False
                set_of_points_to_check_between = [(x,y) for x in start for y in destination]
                #print("green points", green_points)
                #print("orange points", orange_points)
                #print("player_pionts", player_points)
                #print("set of", set_of_points_to_check_between)
                for points in set_of_points_to_check_between:
                    if between:
                        break
                    for other_point in other_points:
                        if self.is_blocking(points[0],points[1],other_point):
                            between = True
                            break
        #     print("BETWEEN", between)
        #
        # history_dict = {'green':green_beacon,'orange':orange_beacon,'blocking':between,
        #                 'actr':False, 'chosen_action':'select_beacon'.upper()}
        #self.history.append(dict(history_dict))
        if target is None:
            obs_dict = {'green':green_beacon, 'orange': orange_beacon}
        else:
            obs_dict = {'green':green_beacon, 'orange':orange_beacon, 'blocking':between}
        return obs_dict

    def decision(self, obs):
        '''Given 1 observation, decide on a plan.'''
        #regardless of decision, the first thing is to click on the player
        player_xys = self.get_player_coords(obs)[0]
        player_xys = np.reshape(np.asarray([player_xys[0],player_xys[1]],dtype=int),(1,2))
        click_player = self.action_processer.process(self.select_army,player_xys)
        useless_obs_raw = self.env.step(click_player)
        useless_obs = self.obs_processer.process(useless_obs_raw)
        green_beacon_xys = self.get_green_beacon_coords(useless_obs)
        orange_beacon_xys = self.get_orange_beacon_coords(useless_obs)


        obs_dict = self.create_obs_dict(useless_obs)

        if green_beacon_xys and orange_beacon_xys:
            best_target = self.memory.best_blend('value',(-1,0),'action')

            #if best_target[0]['action'] == -1:
            if best_target[0] == -1:
                obs_dict = self.create_obs_dict(useless_obs,target='green')
                coords = green_beacon_xys
            #elif best_target[0]['action'] == 0:
            elif best_target[0] == 0:
                obs_dict = self.create_obs_dict(useless_obs,target='orange')
                coords = orange_beacon_xys
        elif green_beacon_xys and not orange_beacon_xys:
            coords = green_beacon_xys
        elif not green_beacon_xys and orange_beacon_xys:
            coords = orange_beacon_xys


        # location_coords = np.reshape(np.asarray([20,20],dtype=int),(1,2))
        done = False
        while not done:
            location_coords = random.choice(coords)
            location_coords = np.reshape(np.asarray([location_coords[0],location_coords[1]],dtype=int),(1,2))
            try:
                click_screen = self.action_processer.process(self.patrol_screen,location_coords)
                useless_obs_raw = self.env.step(click_screen)

            except ValueError:
                useless_obs = self.obs_processer.process(useless_obs_raw)
                player_xys = self.get_player_coords(useless_obs)[0]
                player_xys = np.reshape(np.asarray([player_xys[0], player_xys[1]], dtype=int), (1, 2))
                click_player = self.action_processer.process(self.select_army, player_xys)
                useless_obs_raw = self.env.step(click_player)


            print(useless_obs_raw[-1].reward,useless_obs_raw[-1].last())
            if useless_obs_raw[-1].reward:
                done = True
            if useless_obs_raw[-1].last():
                done = True

        if green_beacon_xys and orange_beacon_xys:
            #return {**obs_dict, **best_target[0], 'value':useless_obs_raw[-1].reward}
            return {**obs_dict, 'action':best_target[0], 'value':useless_obs_raw[-1].reward}
        elif green_beacon_xys and not orange_beacon_xys:
            return {**obs_dict, 'action': -1, 'value':useless_obs_raw[-1].reward}
        elif not green_beacon_xys and orange_beacon_xys:
            return {**obs_dict, 'action':0, 'value':useless_obs_raw[-1].reward}
            # for t in useless_obs_raw:
            #     if t.last():
            #         self.episode_counter += 1
            #         done = True



    def get_green_beacon_coords(self,obs):
        player_relative = obs["player_relative_screen"]
        # neutral is green beacon in this mission
        neutral_x, neutral_y = (player_relative == _PLAYER_NEUTRAL).nonzero()[1:3]
        green_points = []
        if neutral_y.any():
            green_points = list(zip(neutral_x, neutral_y))
        return green_points

    def get_orange_beacon_coords(self,obs):
        player_relative = obs["player_relative_screen"]
        #enemey is orange beacon in this mission
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()[1:3]
        orange_points = []
        if enemy_y.any():
            orange_points = list(zip(enemy_x, enemy_y))
        return orange_points

    def get_player_coords(self,obs):
        player_relative = obs["player_relative_screen"]
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()[1:3]
        player_points = list(zip(player_x, player_y))
        return player_points

    def is_blocking(self,seg1,seg2,point):
        '''if green=point is between player=seg1 and orange=seg2'''
        #https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment
        #print("is_blocking:", seg1, seg2, point)
        crossproduct = (point[1] - seg1[1]) * (seg2[0] - seg1[0]) - (point[0] - seg1[0]) * (seg2[1] - seg1[1])
        if abs(crossproduct) != 0:
            return False

        dotproduct = (point[0] - seg1[0]) * (seg2[0] - seg1[0]) + (point[1] - seg1[1]) * (seg2[1] - seg1[1])
        if dotproduct < 0:
            return False

        squaredlengthba = (seg2[0] - seg1[0]) * (seg2[0] - seg1[0]) + (seg2[1] - seg1[1]) * (seg2[1] - seg1[1])
        if dotproduct > squaredlengthba: return False

        return True


    def select_green(self):
        pass

    def select_orange(self):
        pass

    def reset(self):
        obs = self.env.reset()
        self.latest_obs = self.obs_processer.process(obs)
        return self.latest_obs






