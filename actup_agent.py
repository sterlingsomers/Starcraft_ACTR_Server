from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from pysc2.lib import features
from pysc2.lib import actions
from absl import flags
import numpy as np
import random
import pyactup
import math
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

    def __init__(self,env,noise=0.0,decay=0.0,temperature=1.0,threshold=-100.0,mismatch=1,optimized_learning=False):
        self.memory = pyactup.Memory(noise=noise,decay=decay,temperature=temperature,threshold=threshold,mismatch=mismatch,optimized_learning=optimized_learning)#noise,decay,temperature,threshold,mismatch)
        self.mismatch = mismatch
        self.temperature = temperature
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.env = env
        self.cumulative_score = 0
        self.select_army = np.asarray([7],dtype=int)
        self.patrol_screen = np.asarray([333],dtype=int)
        self.move_screen = np.asarray([331],dtype=int)
        self.noop = np.asarray([0],dtype=int)
        self.episode_counter = 0
        self.step_caught = 0
        self.step = 0
        self.data = {}
        # self.action_map = {-1:self.select_green,0:self.select_orange}
        self.player_selected = False
        #actions
        #1 green
        #0  orange
        # self.memory.learn(green=1,orange=1,action=-1,value=1000)
        # self.memory.learn(green=1,orange=1,action=0,value=1000)
        self.memory.learn(green=1,orange=1,target=0,value=5)
        self.memory.learn(green=1,orange=1,target=1,value=5)
        self.memory.learn(orange=1, green=1, target=1, blocking=0, action=0)
        self.memory.learn(orange=1, green=1, target=1, blocking=1, action=1)
        self.memory.learn(orange=1, green=1, target=0, blocking=0, action=0)
        self.memory.learn(orange=1, green=1, target=0, blocking=1, action=1)

        # self.memory.learn(green=1,orange=1,target=1,blocking=1,action=1)
        # self.memory.learn(green=1,orange=1,target=1,blocking=0,action=0)
        # self.memory.learn(green=1,orange=1,target=0,blocking=0,action=0)
        # self.memory.learn(green=1,orange=1,target=0,blocking=1,action=1)
        # self.memory.learn(green=1,orange=0,target=1,blocking=0,action=0)
        # self.memory.learn(green=0,orange=1,target=0,blocking=0,action=0)

        pyactup.set_similarity_function(self.custom_similarity,*['green','orange','target','blocking'])





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
                between = 0
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
                            between = 1
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

    def compute_S(self,probe, feature_list, history, Vk, MP, t):
        chunk_names = []

        PjxdSims = {}
        PI = math.pi
        for feature in feature_list:
            Fk = probe[feature]
            for chunk in history:
                #I added this check to see if there is a retrieval probabily
                #If not, the chunk should not be included - it wasn't retrieved
                if not 'retrieval_probability' in chunk.keys():
                    continue
                dSim = None
                vjk = None
                for attribute in chunk['attributes']:
                    if attribute[0] == feature:
                        vjk = attribute[1]
                        break

                if Fk == vjk:
                    dSim = 0.0
                else:
                    if 'rads' in feature:
                        a_result = np.argmin(((2 * PI) - abs(vjk-Fk), abs(vjk-Fk)))
                        if not a_result:
                            dSim = (vjk - Fk) / abs(Fk - vjk)
                        else:
                            dSim = (Fk - vjk) / abs(Fk - vjk)
                    else:
                        dSim = (vjk - Fk) / abs(Fk - vjk)

                # if Fk == vjk:
                #     dSim = 0
                # else:
                #     dSim = -1 * ((Fk-vjk) / math.sqrt((Fk - vjk)**2))

                Pj = chunk['retrieval_probability']
                if not feature in PjxdSims:
                    PjxdSims[feature] = []
                PjxdSims[feature].append(Pj * dSim)
                pass

        # vio is the value of the output slot
        fullsum = {}
        result = {}  # dictionary to track feature
        Fk = None
        for feature in feature_list:
            Fk = probe[feature]
            if not feature in fullsum:
                fullsum[feature] = []
            inner_quantity = None
            Pi = None
            vio = None
            dSim = None
            vik = None
            for chunk in history:
                if not 'retrieval_probability' in chunk.keys():
                    continue
                Pi = chunk['retrieval_probability']
                for attribute in chunk['attributes']:
                    if attribute[0] == Vk:
                        vio = attribute[1]

                for attribute in chunk['attributes']:
                    if attribute[0] == feature:
                        vik = attribute[1]
                # if Fk > vik:
                #     dSim = -1
                # elif Fk == vik:
                #     dSim = 0
                # else:
                #     dSim = 1
                # dSim = (Fk - vjk) / sqrt(((Fk - vjk) ** 2) + 10 ** -10)
                if Fk == vik:
                    dSim = 0.0
                else:
                    #dSim = (vik - Fk) / abs(Fk - vik)
                    if 'rads' in feature:
                        a_result = np.argmin(((2 * PI) - abs(vjk-Fk), abs(vjk-Fk)))
                        if not a_result:
                            dSim = (vik - Fk) / abs(Fk - vik)
                        else:
                            dSim = (Fk - vjk) / abs(Fk - vik)
                    else:
                        dSim = (vik - Fk) / abs(Fk - vik)
                #
                # if Fk == vik:
                #     dSim = 0
                # else:
                #     dSim = -1 * ((Fk-vik) / math.sqrt((Fk - vik)**2))

                inner_quantity = dSim - sum(PjxdSims[feature])
                fullsum[feature].append(Pi * inner_quantity * vio)

            result[feature] = sum(fullsum[feature])

        # sorted_results = sorted(result.items(), key=lambda kv: kv[1])
        return result

    def custom_similarity(self,x,y):
        result = 1 - abs(x-y)
        return result

    def decision(self, obs_raw):
        '''Given 1 observation, decide on a plan.'''
        #regardless of decision, the first thing is to click on the player
        print("decision called")
        self.memory.advance()
        err = ''
        self.decision_obs_raw = obs_raw
        time_done = False
        # time.sleep(0.1)
        # noop = self.action_processer.process(self.noop,np.reshape(np.asarray([0,0],dtype=int),(1,2)))
        # obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
        player_xys = self.get_player_coords(obs)#[0]
        if not player_xys:
            print('yup. trouble.')
        player_xys = np.reshape(np.asarray([player_xys[0],player_xys[1]],dtype=int),(1,2))

        click_player = self.action_processer.process(self.select_army,player_xys)
        obs_raw = self.env.step(click_player)
        if obs_raw[0].last():
            print('last caught decision')
            self.episode_counter += 1
            self.cumulative_score = 0
            obs = self.obs_processer.process(obs_raw)
            obs_dict = self.create_obs_dict(obs)
            return {**obs_dict, 'blocking':-1,'target':-1,'value':-1},obs_raw, True, 'time done in decision', {}, {}

        obs = self.obs_processer.process(obs_raw)

        green_beacon_xys = self.get_green_beacon_coords(obs)
        orange_beacon_xys = self.get_orange_beacon_coords(obs)


        obs_dict = self.create_obs_dict(obs)
        self.first_obs_dict = obs_dict.copy()
        print('obs dict', obs_dict)

        value_saliences = []
        if green_beacon_xys and orange_beacon_xys:
            #Only do the blend if there's an option. Otherwise just click on what's available"
            targets = [0, 1]
            target_values = {0:0, 1:0}

            for target in targets:

                self.memory.activation_history = []
                value_blend = self.memory.blend('value',green=1,orange=1,target=target)
                salience = self.compute_S({'green':1, 'orange':1,'target':target},['green','orange','target'],
                                          self.memory.activation_history,'target',self.mismatch,self.temperature)
                salience['value_estimate'] = value_blend
                salience['imagined_target'] = target
                value_saliences.append(salience.copy())
                target_values[target] = value_blend
                #print('here')

            #m_best = self.memory.best_blend('value',(1,0),'target')
            best_target_value = max(target_values.values())#self.memory.best_blend('value',(1,0),'target')
            best_targets = [x for x in target_values if target_values[x] == best_target_value]
            best_target = random.choice(best_targets)
            self.best_target = best_target

            #if best_target[0]['action'] == -1:
            if best_target == 1:
                obs_dict = self.create_obs_dict(obs,target='green')
                obs_dict['target'] = 1
                coords = green_beacon_xys
                obstacle_coords = orange_beacon_xys
            #elif best_target[0]['action'] == 0:
            elif best_target == 0:
                obs_dict = self.create_obs_dict(obs,target='orange')
                obs_dict['target'] = 0
                coords = orange_beacon_xys
                obstacle_coords = green_beacon_xys
        elif green_beacon_xys and not orange_beacon_xys:
            coords = green_beacon_xys
            obs_dict['target'] = 1
            obs_dict['blocking'] = 0
        elif not green_beacon_xys and orange_beacon_xys:
            obs_dict['target'] = 0
            obs_dict['blocking'] = 0
            coords = orange_beacon_xys

        #At this point we've picked a target.
        #and have a new blend to determine if the target is blocked
        #Now let's determine action: go-to-target or go-around-obstacle
        action = 0 #go direct is default until you make a choice
        action_salience = {'green':np.NaN,'orange':np.NaN,'target':np.NaN,'blocking':np.NaN}
        if green_beacon_xys and orange_beacon_xys:
            self.memory.activation_history = []
            action = self.memory.blend('action', **obs_dict)
            salience = self.compute_S(obs_dict, ['green', 'orange', 'target', 'blocking'],
                                      self.memory.activation_history, 'action', self.mismatch, self.temperature)
            action_salience = dict(**salience)
        #print('here')
        #action = self.memory.retrieve(partial=True,**obs_dict)['action']
        action_salience['action'] = action
        action = round(action)
        #obs_dict['action'] = action

        print(obs_dict)
        if action: #i.e. go around
            reward,obs,time_done,err = self.go_around(obs_raw, target_coords=coords,obstacle_coords=obstacle_coords,target_val=obs_dict['target'])
        elif action == 0 and obs_dict['target'] == 1: #go directly to green
            reward,obs,time_done,err = self.select_beacon(obs_raw,obs_dict['target'],chunk=obs_dict)
        elif action == 0 and obs_dict['target'] == 0: #go directly to orange
            reward,obs,time_done,err = self.select_beacon(obs_raw,obs_dict['target'],chunk=obs_dict)


        return {**obs_dict, 'value':reward},obs_raw,time_done,err,value_saliences,action_salience


    def get_around_coords(self,obs,target_coords,obstacle_coords):
        pass



    def get_green_beacon_coords(self,obs):
        # noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        # obs_raw = self.env.step(noop)
        # obs = self.obs_processer.process(obs_raw)
        player_relative = obs["player_relative_screen"]
        # neutral is green beacon in this mission
        neutral_x, neutral_y = (player_relative == _PLAYER_NEUTRAL).nonzero()[1:3]
        green_points = []
        if neutral_y.any():
            green_points = list(zip(neutral_x, neutral_y))
        else:
            print('problem - green')
        return green_points

    def get_orange_beacon_coords(self,obs):
        # noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        # obs_raw = self.env.step(noop)
        # obs = self.obs_processer.process(obs_raw)
        player_relative = obs["player_relative_screen"]
        #enemey is orange beacon in this mission
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()[1:3]
        orange_points = []
        if enemy_y.any():
            orange_points = list(zip(enemy_x, enemy_y))
        else:
            print('problem - orange')
        return orange_points

    def get_player_coords(self,obs):
        # noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        # obs_raw = self.env.step(noop)
        # obs = self.obs_processer.process(obs_raw)
        player_relative = obs["player_relative_screen"]
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()[1:3]
        if player_y.any():
            player_points = list(zip(player_x, player_y))
        else:
            print("PROBLEM!")
            return []
        if len(player_points) > 1:
            return random.choice(player_points)
        return player_points[0]

    def go_around(self,obs_raw,target_coords=[],obstacle_coords=[],target_val=0):
        print("go around called")
        err = ''
        target_rewards = {0:50, 1:1}
        # noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        # obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
        target_map = {1:'green', 0:'orange'}
        player_xys = self.get_player_coords(obs)#[0]

        if self.cumulative_score > obs_raw[0].observation['score_cumulative'][0] and obs_raw[0].observation['score_cumulative'][0] == 0:
            #retting the score probably didn't go right
            self.cumulative_score = 0

        print("go around target", target_val)
        # alt_player = np.where(obs['player_relative_screen'] == 1)
        # alt_player = alt_player[1:]
        # alt_player = (alt_player[0][0],alt_player[1][0])
        # # zero_points = np.where(obs['player_relative_screen'] == 0)
        # # zero_points = list(zip(zero_points[0],zero_points[1]))

        # if not obs['player_relative_screen'][:,player_xys[0],player_xys[1]][0]:
        #     print('player position wrong...')



        #find the center point of each n_coords
        # centroid_player_xs = [p[0] for p in player_xys]
        # centroid_player_ys = [p[1] for p in player_xys]
        # player_centroid = (sum(centroid_player_xs)/len(player_xys), sum(centroid_player_ys) / len(player_xys))
        #Player is only 1 point - so there is no need to determine the centroid.

        centroid_target_xs = [p[0] for p in target_coords]
        centroid_target_ys = [p[1] for p in target_coords]
        # target_centroid = (int(round(sum(centroid_target_xs)/len(target_coords))), int(round(sum(centroid_target_ys) / len(target_coords))))
        #that centroid calculation does not work.
        #try alternate
        target_centroid = (max(centroid_target_xs) + min(centroid_target_xs)) / 2, (max(centroid_target_ys) + min(centroid_target_ys)) / 2

        centroid_obstacle_xs = [p[0] for p in obstacle_coords]
        centroid_obstacle_ys = [p[1] for p in obstacle_coords]
        obstacle_centroid = (max(centroid_obstacle_xs) + min(centroid_obstacle_xs)) / 2, (max(centroid_obstacle_ys) + min(centroid_obstacle_ys)) / 2

        distances = [0]
        d = 2
        while min(distances) < 5:
            d += 0.5
            a1 = player_xys[0]
            a2 = player_xys[1]
            b1 = obstacle_centroid[0]
            b2 = obstacle_centroid[1]

            if b1 - a1 < 0.001:
                a1 += 0.001

            y1 = b2 + (((d ** 2) / (1 + (((b2 - a2) / (b1 - a1)) ** 2))) ** 1 / 2)
            y2 = b2 - (((d**2)/(1+(((b2-a2)/(b1-a1))**2)) )**1/2)

            x1 = (((b2-a2)/(b1-a1)) * (b2-y1)) + b1
            x2 = (((b2-a2)/(b1-a1)) * (b2-y2)) + b1

            p1 = int(round(x1)), int(round(y1))
            p2 = int(round(x2)), int(round(y2))

            distances = [np.linalg.norm(np.array(p)-np.array(p1)) for p in obstacle_coords]

        # ps = [p1,p2]
        # p_dist = {p:np.linalg.norm(np.array(p)-np.array(target_centroid)) for p in ps}
        # p_sorted = {k:v for k,v in sorted(p_dist.items(), key=lambda item: item[1])}
        p1_distance = np.linalg.norm(np.array(p1)-np.array(target_centroid))
        p2_distance = np.linalg.norm(np.array(p2)-np.array(target_centroid))
        if p1_distance < p2_distance:
            around1 = np.reshape(np.asarray(p1,dtype=int), (1,2))
            around2 = np.reshape(np.asarray(p2,dtype=int), (1,2))
        else:
            around1 = np.reshape(np.asarray(p2, dtype=int), (1, 2))
            around2 = np.reshape(np.asarray(p1, dtype=int), (1, 2))

        around_point = around1
        if around_point[0][0] >= flags.FLAGS.resolution or around_point[0][0] <= 0 or around_point[0][1] >= flags.FLAGS.resolution or around_point[0][1] <= 0:
            around_point = around2


        # around1 = np.reshape(np.asarray([p_], dtype=int), (1, 2))
        # around2 = np.reshape(np.asarray([x2, y2], dtype=int), (1, 2))




        # print('here...')





        #goal_rads = math.atan2(goal_location[0] - agent_location[0], goal_location[1] - agent_location[1])

        #now do the equalateral triangle bits
        #https://stackoverflow.com/questions/50547068/creating-an-equilateral-triangle-for-given-two-points-in-the-plane-python
        # M = (player_xys[0]+target_centroid[0])/2, (player_xys[1]+target_centroid[1])/2
        # O = (player_xys[0]-M[0])*3**0.5, (player_xys[1]-M[1])*3**0.5
        #
        # p1 = M[0]+O[1], M[1]-O[0]
        # p2 = M[0]-O[1], M[1]+O[0]
        # around_point = np.reshape(np.asarray([p1[0], p1[1]], dtype=int), (1, 2))
        #
        # if p1[0] > 32 or p1[0] < 0 or p1[1] > 32 or p1[1] < 0:
        #     if p2[0] > 32 or p2[0] < 0 or p2[1] > 32 or p2[1] < 0:
        #        print("both around points are bad.")
        #        return 0, self.reset()
        #
        #
        #     else:
        #         around_point = np.reshape(np.asarray([p2[0], p2[1]], dtype=int), (1, 2))
        #https: // math.stackexchange.com / questions / 3177129 / find - third - point - in -right - triangle - given - two - points - and -a - length
        # d = 5
        #         # u = player_xys[0]
        #         # v = player_xys[1]
        #         # w = obstacle_centroid[0]
        #         # z = obstacle_centroid[1]
        #         #
        #         # around_point = (d/((1+(((z-v)/(w-u))**2))**1/2)) + u, ((-1*(d*(w-u)))/((z-v)*((1 + (((z-v)/(w-u))**2))**1/2)) ) + v
        #         # around_point = int(round(around_point[0])),int(round(around_point[1]))
        #         # around_point = np.reshape(np.asarray([around_point[0], around_point[1]], dtype=int), (1, 2))

        ####
        # p1,p2 = obstacle_coords[0],obstacle_coords[0]
        # d = 2
        # while p1 in obstacle_coords and p2 in obstacle_coords:
        #     d += 3.25
        #     a1 = player_xys[0]
        #     a2 = player_xys[1]
        #     b1 = obstacle_centroid[0]
        #     b2 = obstacle_centroid[1]
        #
        #     if b1 - a1 < 0.001:
        #         a1 += 0.001
        #
        #     y1 = b2 + (((d**2)/(1+(((b2-a2)/(b1-a1))**2)) )**1/2)
        #     y2 = b2 - (((d**2)/(1+(((b2-a2)/(b1-a1))**2)) )**1/2)
        #
        #     x1 = (((b2-a2)/(b1-a1)) * (b2-y1)) + b1
        #     x2 = (((b2-a2)/(b1-a1)) * (b2-y2)) + b1
        #
        #     try:
        #
        #         p1 = int(round(x1)),int(round(y1))
        #     except ValueError:
        #         print('troubles...debug here')
        #     try:
        #         p2 = int(round(x2)),int(round(y2))
        #     except ValueError:
        #         print('troubles2 debug ere')
        #
        # around1 = np.reshape(np.asarray([x1, y1], dtype=int), (1, 2))
        # around2 = np.reshape(np.asarray([x2, y2], dtype=int), (1, 2))
        #
        # around_point = around1
        #
        #
        #
        #
        done = False
        bad_end = False
        previous_player_xys = (-1,-1)
        VerrCount = 0
        while not done:
            #first check that we didn't hit a beacon or time has ended
            if obs_raw[-1].reward:
                self.cumulative_score += obs_raw[0].reward
                reward = obs_raw[0].reward
                while not obs_raw[0].observation['score_cumulative'][0] == self.cumulative_score:
                    noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                    obs_raw = self.env.step(noop).copy()
                    if obs_raw[0].last():
                        print('damn...')

                self.episode_counter += 1
                # if not obs_raw[-1].reward == target_rewards[target_val]:
                #     reward = -1
                #     err = 'hit beacon by accident (go around)'
                #     bad_end = True
                # else:
                err = 'reward - go around'
                reward = obs_raw[-1].reward
                return reward,obs_raw,bad_end,err



            if obs_raw[-1].last():
                self.cumulative_score = 0
                noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                updated_obs_raw = self.env.step(noop)
                updated_obs = self.obs_processer.process(obs_raw)
                if updated_obs_raw[0].last():
                    print('damn...')
                self.episode_counter += 1
                bad_end = True
                err = 'last - time'
                return -1, updated_obs_raw, bad_end, err

            if not obs_raw[0].observation['score_cumulative'][0] == self.cumulative_score:
                reward = obs_raw[0].observation['score_cumulative'][0] - self.cumulative_score
                loops = 0
                while not obs_raw[0].reward == reward:
                    noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                    obs_raw = self.env.step(noop).copy()
                    if obs_raw[0].last():
                        print('damn...')
                    loops += 1
                    if loops > 10:
                        break
                self.episode_counter += 1

                err = 'reward'
                self.cumulative_score += reward
                # if not reward == target_rewards[target_val]:
                #     reward = -1
                #     bad_end = True
                #     err = 'hit wrong beacon by accident (go_around 2)'
                return reward,obs_raw,bad_end,err




            if 333 in obs_raw[0].observation['available_actions']:
                try:
                    click_screen = self.action_processer.process(self.patrol_screen, around_point)
                    obs_raw = self.env.step(click_screen)
                    if obs_raw[0].last():
                        self.cumulative_score = 0
                        self.episode_counter += 1
                        err = 'time - around 2'
                        noop = self.action_processer.process(self.noop,
                                                             np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                        updated_obs_raw = self.env.step(noop)
                        updated_obs = self.obs_processer.process(obs_raw)
                        if updated_obs_raw[0].last():
                            print('damn...')
                        return -1,obs_raw,True,err

                except ValueError as err:
                    print(err)
                    print("should not get here with new if statements...")
                except AssertionError as err:
                    print('err', around_point)
            else:
                try:
                    player_to_click = np.reshape(np.asarray([player_xys[0], player_xys[1]], dtype=int), (1, 2))
                    click_player = self.action_processer.process(self.select_army, player_to_click)
                    obs_raw = self.env.step(click_player)
                    if obs_raw[0].last():
                        print('damn...')
                except ValueError as err2:
                    print(err2)
                    print("should not get ere either, with new if statements")
                continue






            #try clicking to the around point
            # try:
            #
            #     click_screen = self.action_processer.process(self.patrol_screen, around_point)
            #     obs_raw = self.env.step(click_screen)
            #
            # #If there's a valueError - it's probably because the agent isn't selected
            # except ValueError as err:
            #     print(err)
            #     if 333 in obs_raw[0].observation['available_actions']:
            #         #that means the problem is an INDEX error (clicking off screen)
            #         around_point = around2
            #         continue
            #     try:
            #         click_screen = self.action_processer.process(self.move_screen, around_point)
            #         obs_raw = self.env.step(click_screen)
            #     except ValueError as err2:
            #         print(err2)
            #         try:
            #             player_xys = self.get_player_coords(obs)#[0]
            #         except IndexError as err3:
            #             #If we cannot select the player, this is a bad end
            #             print(err3)
            #             noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
            #             updated_obs_raw = self.env.step(noop)
            #             updated_obs = self.obs_processer.process(obs_raw)
            #             self.episode_counter += 1
            #             bad_end = True
            #             return -1, updated_obs_raw, bad_end
            #
            #         player_to_click = np.reshape(np.asarray([player_xys[0], player_xys[1]], dtype=int), (1, 2))
            #         click_player = self.action_processer.process(self.select_army, player_to_click)
            #         obs_raw = self.env.step(click_player)
            #         print('go around - reselect player')
            #         VerrCount += 1
            #         if VerrCount > 3:
            #             #it's not letting me move...
            #             noop = self.action_processer.process(self.noop,
            #                                                  np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
            #             updated_obs_raw = self.env.step(noop)
            #             updated_obs = self.obs_processer.process(obs_raw)
            #             self.episode_counter += 1
            #             bad_end = True
            #             return -1, updated_obs_raw, bad_end




            # except AssertionError:
            #     around_point = around2
            #     continue

            obs = self.obs_processer.process(obs_raw)
            obs_dict = self.create_obs_dict(obs,target=target_map[target_val])
            if not obs_dict['blocking']:
                done = True
                break
            try:
                player_xys = self.get_player_coords(obs)
            except IndexError:
                # If we cannot select the player, this is a bad end
                noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                updated_obs_raw = self.env.step(noop)
                if updated_obs_raw[0].last():
                    print('damn...')
                updated_obs = self.obs_processer.process(obs_raw)
                self.episode_counter += 1
                bad_end = True
                err = 'cannot click player'
                return -1, updated_obs_raw, bad_end, err

            if previous_player_xys == player_xys:
                #That means we haven't moved
                return self.go_around(obs_raw,target_coords,obstacle_coords,target_val)

            previous_player_xys = player_xys




        return self.select_beacon(obs_raw,target_val)


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

    def neighbors(self,x,y,anArray):
        X = anArray.shape[0]
        Y = anArray.shape[1]
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2)
                     for y2 in range(y - 1, y + 2)
                     if (-1 < x <= X and
                         -1 < y <= Y and
                         (x != x2 or y != y2) and
                         (0 <= x2 <= X) and
                         (0 <= y2 <= Y))]
        return neighbors

    def select_beacon(self,obs_raw,target_val,chunk=None):
        print("select beacon called")
        if obs_raw[0].last():
            err = 'last'
            print("last caught select beacon 1")
            return -1, obs_raw, True, err
        if not int(obs_raw[0].step_type):
            err = 'first'
            print("first frame caught - last epsidoe ended")
            return -1, obs_raw, True, err
        # previous_obs = []
        # previous_obs.append(obs_raw.copy())
        print("select beacon target", target_val)
        err = ''
        target_rewards = {0:50, 1:1}
        # noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        # obs_raw = self.env.step(noop).copy()
        obs = self.obs_processer.process(obs_raw)
        # previous_obs.append(obs_raw.copy())
        target_map = {1: self.get_green_beacon_coords, 0: self.get_orange_beacon_coords}
        coords = target_map[target_val](obs)
        done = False
        time_done = False
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]

        if self.cumulative_score > obs_raw[0].observation['score_cumulative'][0] and obs_raw[0].observation['score_cumulative'][0] == 0:
            #retting the score probably didn't go right
            self.cumulative_score = 0
        #centroid = (sum(xs) / len(coords), sum(ys) / len(coords))
        #centroid = np.reshape(np.asarray([centroid[0],centroid[1]],dtype=int),(1,2))
        #the centroid is not always sufficient, it seems.
        #center = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
        #center = np.reshape(np.asarray([center[0], center[1]], dtype=int), (1, 2))
        # point = random.choice(coords)
        # point = np.reshape(np.asarray([point[0], point[1]], dtype=int), (1, 2))
        reward = 0
        print("select beacon while loop about to start")
        while not done:
            coords = target_map[target_val](obs)
            try:
                point = random.choice(coords)
            except IndexError as err:
                #if there are no points, then it's switched maps BEFORE getting the reward - we need to keep doing noop
                #until the reward comes through
                noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                obs_raw = self.env.step(noop).copy()
                print('debug... 3')
                # if obs_raw[0].last():
                #     print('damn...')
                # print('debug')

                # print("problem - set debug point")
                # err = err.__str__() + ' - due to wrong obs'
                # return -1, obs_raw, True, err
            else:
                point = np.reshape(np.asarray([point[0], point[1]], dtype=int), (1, 2))
                try:
                    # previous_obs.append(obs_raw.copy())
                    click_screen = self.action_processer.process(self.patrol_screen,point)
                    obs_raw = self.env.step(click_screen).copy()
                    print('debug... 4')
                    # if obs_raw[0].last():
                    #     self.cumulative_score = 0
                    #     self.episode_counter += 1
                    #     noop = self.action_processer.process(self.noop,
                    #                                          np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                    #     updated_obs_raw = self.env.step(noop)
                    #     updated_obs = self.obs_processer.process(obs_raw)
                    #     return -1,updated_obs_raw,True,'time up - select beacon 2'

                except ValueError:
                    noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                    obs_raw = self.env.step(noop).copy()
                    print('debug... 5')
                    # if obs_raw[0].last():
                    #     print('damn...')
                    # print('debug')
                # previous_obs = obs_raw.copy()
                # obs = self.obs_processer.process(obs_raw)
                # player_xys = self.get_player_coords(obs)#[0]
                # player_xys = np.reshape(np.asarray([player_xys[0], player_xys[1]], dtype=int), (1, 2))
                # click_player = self.action_processer.process(self.select_army, player_xys)
                # obs_raw = self.env.step(click_player).copy()

            # print(obs_raw[0].last(), obs_raw[0].reward, obs_raw[0].observation['score_cumulative'][0])

            # print(obs_raw[-1].reward,obs_raw[-1].last())
            if self.cumulative_score > obs_raw[0].observation['score_cumulative'][0] and \
                    obs_raw[0].observation['score_cumulative'][0] == 0:
                # retting the score probably didn't go right
                self.cumulative_score = 0

            if obs_raw[-1].reward:
                self.cumulative_score += int(obs_raw[-1].reward)
                reward = obs_raw[-1].reward
                while not obs_raw[0].observation['score_cumulative'][0] == self.cumulative_score:
                    noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                    obs_raw = self.env.step(noop).copy()
                    print('debug... 6')
                    if obs_raw[0].last():
                        print('damn... select beacon 2')

                done = True
                self.episode_counter += 1
                err = 'reward'
                #cannot do following test with stochastic reward
                # if not reward == target_rewards[target_val]:
                #     reward = -1
                #     err = 'hit wrong beacon by accident (select_beacon)'
                break



            if obs_raw[-1].last():
                done = True
                self.cumulative_score = 0
                self.episode_counter += 1
                #below might fix the problem where it doesn't reset properly
                time_done = True
                reward = -1
                err = 'last'
                print('last caught select beacon 2')
                break

            if not int(obs_raw[0].observation['score_cumulative'][0]) == int(self.cumulative_score):
                reward = int(obs_raw[0].observation['score_cumulative'][0] - self.cumulative_score)
                loops = 0
                while not int(obs_raw[0].reward) == int(reward):
                    noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                    obs_raw = self.env.step(noop).copy()
                    print('debug... 1')
                    if obs_raw[0].last():
                        print('damn...')
                    loops += 1
                    if loops > 10:
                        break
                    # reward = obs_raw[0].observation['score_cumulative'][0] - self.cumulative_score

                self.episode_counter += 1
                done = True

                err = 'reward'
                self.cumulative_score += reward
                #I cannot do the following test with stochastic reward.
                # if not int(reward) == int(target_rewards[target_val]):
                #     reward = -1
                #     err = 'hit wrong beacon by accident (select_becon 2)'

                break

        noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        updated_obs_raw = self.env.step(noop)
        updated_obs = self.obs_processer.process(obs_raw)
        print('debug... 2')
        print('select beacon done')
        return reward, obs_raw, time_done, err



    def select_orange(self):
        pass

    def reset(self):
        obs = self.env.reset()
        # self.latest_obs = self.obs_processer.process(obs)
        #return self.latest_obs
        return obs






