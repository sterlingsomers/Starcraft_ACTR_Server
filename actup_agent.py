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

    def __init__(self,env,noise=0.2,decay=0.5,temperature=1.0,threshold=-100.0,mismatch=1.5,optimized_learning=False):
        self.memory = pyactup.Memory(noise,decay,temperature,threshold,mismatch)
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.env = env
        self.select_army = np.asarray([7],dtype=int)
        self.patrol_screen = np.asarray([333],dtype=int)
        self.noop = np.asarray([0],dtype=int)
        self.episode_counter = 0
        # self.action_map = {-1:self.select_green,0:self.select_orange}
        self.player_selected = False
        #actions
        #1 green
        #0  orange
        # self.memory.learn(green=1,orange=1,action=-1,value=1000)
        # self.memory.learn(green=1,orange=1,action=0,value=1000)
        self.memory.learn(green=0,orange=1,target=0,value=100)
        self.memory.learn(green=1,orange=0,target=1,value=100)
        self.memory.learn(green=1,orange=1,target=1,blocking=1,action=1)
        self.memory.learn(green=1,orange=1,target=1,blocking=0,action=0)
        self.memory.learn(green=1,orange=1,target=0,blocking=0,action=0)
        self.memory.learn(green=1,orange=1,target=0,blocking=1,action=1)



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

    def decision(self, obs):
        '''Given 1 observation, decide on a plan.'''
        #regardless of decision, the first thing is to click on the player
        self.first_obs = obs
        time_done = False
        # time.sleep(0.1)
        noop = self.action_processer.process(self.noop,np.reshape(np.asarray([0,0],dtype=int),(1,2)))
        obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
        player_xys = self.get_player_coords(obs)[0]
        player_xys = np.reshape(np.asarray([player_xys[0],player_xys[1]],dtype=int),(1,2))

        click_player = self.action_processer.process(self.select_army,player_xys)
        obs_raw = self.env.step(click_player)

        obs = self.obs_processer.process(obs_raw)

        green_beacon_xys = self.get_green_beacon_coords(obs)
        orange_beacon_xys = self.get_orange_beacon_coords(obs)


        obs_dict = self.create_obs_dict(obs)

        if green_beacon_xys and orange_beacon_xys:
            #Only do the blend if there's an option. Otherwise just click on what's available"
            best_target = self.memory.best_blend('value',(1,0),'target')
            self.best_target = best_target

            #if best_target[0]['action'] == -1:
            if best_target[0] == 1:
                obs_dict = self.create_obs_dict(obs,target='green')
                obs_dict['target'] = 1
                coords = green_beacon_xys
                obstacle_coords = orange_beacon_xys
            #elif best_target[0]['action'] == 0:
            elif best_target[0] == 0:
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
        action = self.memory.blend('action', **obs_dict)
        #action = self.memory.retrieve(partial=True,**obs_dict)['action']

        action = round(action)

        if action: #i.e. go around
            reward,obs,time_done = self.go_around(obs, target_coords=coords,obstacle_coords=obstacle_coords,target_val=obs_dict['target'])
        if action == 0 and obs_dict['target'] == 1:
            reward,obs,time_done = self.select_beacon(obs_raw,obs_dict['target'])
        if action == 0 and obs_dict['target'] == 0:
            reward,obs,time_done = self.select_beacon(obs_raw,obs_dict['target'])


        return {**obs_dict, 'value':reward},obs,time_done



        #The following loop carries out the entire clicking plan: go-to-green, go-to-orange, or go-around-to-target
        # done = False
        # while not done:
        #     location_coords = random.choice(coords)
        #     location_coords = np.reshape(np.asarray([location_coords[0],location_coords[1]],dtype=int),(1,2))
        #     try:
        #         click_screen = self.action_processer.process(self.patrol_screen,location_coords)
        #         useless_obs_raw = self.env.step(click_screen)
        #
        #     except ValueError:
        #         useless_obs = self.obs_processer.process(useless_obs_raw)
        #         player_xys = self.get_player_coords(useless_obs)[0]
        #         player_xys = np.reshape(np.asarray([player_xys[0], player_xys[1]], dtype=int), (1, 2))
        #         click_player = self.action_processer.process(self.select_army, player_xys)
        #         useless_obs_raw = self.env.step(click_player)
        #
        #
        #     print(useless_obs_raw[-1].reward,useless_obs_raw[-1].last())
        #     if useless_obs_raw[-1].reward:
        #         done = True
        #         self.episode_counter += 1
        #     if useless_obs_raw[-1].last():
        #         done = True
        #
        # if green_beacon_xys and orange_beacon_xys:
        #     #return {**obs_dict, **best_target[0], 'value':useless_obs_raw[-1].reward}
        #     return {**obs_dict, 'target':best_target[0], 'value':useless_obs_raw[-1].reward}
        # elif green_beacon_xys and not orange_beacon_xys:
        #     return {**obs_dict, 'target': 1, 'value':useless_obs_raw[-1].reward}
        # elif not green_beacon_xys and orange_beacon_xys:
        #     return {**obs_dict, 'target': 0, 'value':useless_obs_raw[-1].reward}
        #     # for t in useless_obs_raw:
        #     #     if t.last():
        #     #         self.episode_counter += 1
        #     #         done = True


    def get_around_coords(self,obs,target_coords,obstacle_coords):
        pass



    def get_green_beacon_coords(self,obs):
        noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
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
        noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
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
        noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
        player_relative = obs["player_relative_screen"]
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()[1:3]
        if player_y.any():
            player_points = list(zip(player_x, player_y))
        else:
            print("PROBLEM!")
            return []
        return player_points

    def go_around(self,obs,target_coords=[],obstacle_coords=[],target_val=0):
        noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
        target_map = {1:'green', 0:'orange'}
        player_xys = self.get_player_coords(obs)[0]
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
        p1,p2 = obstacle_coords[0],obstacle_coords[0]
        d = 2
        while p1 in obstacle_coords and p2 in obstacle_coords:
            d += 2.5
            a1 = player_xys[0]
            a2 = player_xys[1]
            b1 = obstacle_centroid[0]
            b2 = obstacle_centroid[1]

            if b1 - a1 < 0.001:
                a1 += 0.001

            y1 = b2 + (((d**2)/(1+(((b2-a2)/(b1-a1))**2)) )**1/2)
            y2 = b2 - (((d**2)/(1+(((b2-a2)/(b1-a1))**2)) )**1/2)

            x1 = (((b2-a2)/(b1-a1)) * (b2-y1)) + b1
            x2 = (((b2-a2)/(b1-a1)) * (b2-y2)) + b1

            try:

                p1 = int(round(x1)),int(round(y1))
            except ValueError:
                print('troubles...debug here')
            try:
                p2 = int(round(x2)),int(round(y2))
            except ValueError:
                print('troubles2 debug ere')

        around1 = np.reshape(np.asarray([x1, y1], dtype=int), (1, 2))
        around2 = np.reshape(np.asarray([x2, y2], dtype=int), (1, 2))

        around_point = around1




        done = False
        old_player_xys = (-1,-1)
        fixme = False
        while not done:
            try:

                click_screen = self.action_processer.process(self.patrol_screen, around_point)
                obs_raw = self.env.step(click_screen)




            except ValueError:
                try:
                    player_xys = self.get_player_coords(obs)[0]
                except IndexError:
                    player_xys = last_player_xys
                player_xys = np.reshape(np.asarray([player_xys[0], player_xys[1]], dtype=int), (1, 2))
                click_player = self.action_processer.process(self.select_army, player_xys)
                obs_raw = self.env.step(click_player)
                last_player_xys = player_xys

            except AssertionError:
                around_point = around2
                continue

            obs = self.obs_processer.process(obs_raw)
            try:
                updated_player_xys = self.get_player_coords(obs)[0]
            except IndexError:
                #probably caused by the player getting too close to another object
                #we'll deal with that by causing it to be caught in a later if, which will nudge the player
                updated_player_xys = player_xys
                fixme = True
            if not fixme:
                obs_dict = self.create_obs_dict(obs,target=target_map[target_val])
                if not obs_dict['blocking']:
                    done = True
                    break

            if updated_player_xys == player_xys:
                #that means we haven't moved.
                # nd_screen = obs['player_relative_screen']
                # nd_screen = nd_screen.reshape((32,32))
                # surrounding = self.neighbors(updated_player_xys[0],updated_player_xys[1],nd_screen)
                # surrounding = [np.array(x) for x in surrounding]
                # distance_by_point = {x:np.linalg.norm()}
                p1, p2 = obstacle_centroid, obstacle_centroid
                d = 2
                while p1 in obstacle_coords and p2 in obstacle_coords:
                    d += 2.75
                    a1 = player_xys[0]
                    a2 = player_xys[1]
                    b1 = obstacle_centroid[0]
                    b2 = obstacle_centroid[1]

                    if b1 - a1 < 0.001:
                        a1 += 0.001

                    y1 = b2 + (((d ** 2) / (1 + (((b2 - a2) / (b1 - a1)) ** 2))) ** 1 / 2)
                    y2 = b2 - (((d ** 2) / (1 + (((b2 - a2) / (b1 - a1)) ** 2))) ** 1 / 2)

                    x1 = (((b2 - a2) / (b1 - a1)) * (b2 - y1)) + b1
                    x2 = (((b2 - a2) / (b1 - a1)) * (b2 - y2)) + b1

                    try:

                        p1 = int(round(x1)), int(round(y1))
                    except ValueError:
                        print('troubles...debug here')
                    try:
                        p2 = int(round(x2)), int(round(y2))
                    except ValueError:
                        print('troubles2 debug ere')
                current_distance = np.linalg.norm(np.array(player_xys) - np.array(obstacle_centroid))
                p1_distance = np.linalg.norm(np.array(p1) - np.array(obstacle_centroid))
                p2_distance = np.linalg.norm(np.array(p2) - np.array(obstacle_centroid))
                if p1_distance > p2_distance:
                    around_point = np.reshape(np.asarray([x2, y2], dtype=int), (1, 2))
                else:
                    around_point = np.reshape(np.asarray([x1, y1], dtype=int), (1, 2))
                print('ere')
                fixme = False
            player_xys = updated_player_xys
            # if int(p1[0]) == int(updated_player_xys[0]) and int(p1[1]) == int(player_xys[1]):
            #     done = True

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

    def select_beacon(self,obs_raw,target_val):
        noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        obs_raw = self.env.step(noop)
        obs = self.obs_processer.process(obs_raw)
        target_map = {1: self.get_green_beacon_coords, 0: self.get_orange_beacon_coords}
        coords = target_map[target_val](obs)
        done = False
        time_done = False
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        #centroid = (sum(xs) / len(coords), sum(ys) / len(coords))
        #centroid = np.reshape(np.asarray([centroid[0],centroid[1]],dtype=int),(1,2))
        #the centroid is not always sufficient, it seems.
        #center = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
        #center = np.reshape(np.asarray([center[0], center[1]], dtype=int), (1, 2))
        # point = random.choice(coords)
        # point = np.reshape(np.asarray([point[0], point[1]], dtype=int), (1, 2))
        while not done:
            print(target_val)
            point = random.choice(coords)
            point = np.reshape(np.asarray([point[0], point[1]], dtype=int), (1, 2))
            try:
                click_screen = self.action_processer.process(self.patrol_screen,point)
                obs_raw = self.env.step(click_screen)

            except ValueError:
                obs = self.obs_processer.process(obs_raw)
                player_xys = self.get_player_coords(obs)[0]
                player_xys = np.reshape(np.asarray([player_xys[0], player_xys[1]], dtype=int), (1, 2))
                click_player = self.action_processer.process(self.select_army, player_xys)
                obs_raw = self.env.step(click_player)


            print(obs_raw[-1].reward,obs_raw[-1].last())
            if obs_raw[-1].reward:
                done = True
                self.episode_counter += 1

            if obs_raw[-1].last():
                done = True
                self.episode_counter += 1
                #below might fix the problem where it doesn't reset properly
                time_done = True

        noop = self.action_processer.process(self.noop, np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
        updated_obs_raw = self.env.step(noop)
        updated_obs = self.obs_processer.process(obs_raw)
        return obs_raw[-1].reward, updated_obs, time_done



    def select_orange(self):
        pass

    def reset(self):
        obs = self.env.reset()
        self.latest_obs = self.obs_processer.process(obs)
        return self.latest_obs






