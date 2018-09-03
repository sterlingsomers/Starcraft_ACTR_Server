import numpy as np


from pysc2.lib import features
from pysc2.lib import actions


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


class episode_filter():

    def __init__(self):
        self.initialized = 0
        self.previous_step = None

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

    def obs_to_dict(self,obs):
        player_relative = obs['player_relative_screen']
        orange_beacon = False
        green_beacon = False
        player = False
        between = False

        neutral_x, neutral_y = (player_relative == _PLAYER_NEUTRAL).nonzero()[1:3]
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()[1:3]
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()[1:3]

        if neutral_y.any():
            green_beacon = True

            if enemy_y.any():
                orange_beacon = True
            if player_y.any():
                player = True

        if enemy_y.any():
            orange_beacon = True

            if neutral_y.any():
                green_beacon = True
            if player_y.any():
                player = True

        print("push_observation:", orange_beacon, player, green_beacon)
        if orange_beacon and player and green_beacon:
            # check for blocking or overlap

            # Determine if green is between orange and player
            green_points = list(zip(neutral_x, neutral_y))
            orange_points = list(zip(enemy_x, enemy_y))
            player_points = list(zip(player_x, player_y))

            # https: // stackoverflow.com / questions / 328107 / how - can - you - determine - a - point - is -between - two - other - points - on - a - line - segment
            between = False
            set_of_points_to_check_between = [(x, y) for x in player_points for y in orange_points]
            # print("green points", green_points)
            # print("orange points", orange_points)
            # print("player_pionts", player_points)
            # print("set of", set_of_points_to_check_between)
            for points in set_of_points_to_check_between:
                for green_point in green_points:
                    if self.is_blocking(points[0], points[1], green_point):
                        between = True
            print("BETWEEN", between)

        chosen_action = 'green_beacon'

        if orange_beacon:
            chosen_action = 'orange_beacon'

        history_dict = {'green': green_beacon, 'orange': orange_beacon, 'blocking': between,
                        'actr': False, 'chosen_action': chosen_action.upper(),
                        'neutral_x': neutral_x, 'neutral_y': neutral_y,
                        'enemy_x': enemy_x, 'enemy_y': enemy_y,
                        'player_x': player_x, 'player_y': player_y}

        return history_dict

    def this_step(self,history_dict):
        if not self.initialized:
            self.initialized = 1
            self.previous_step = history_dict

            return 1
        else:
            current_step = history_dict
            if np.array_equal(current_step['neutral_x'], self.previous_step['neutral_x']) and \
                np.array_equal(current_step['neutral_y'], self.previous_step['neutral_y']) and \
                np.array_equal(current_step['enemy_x'], self.previous_step['enemy_x']) and \
                np.array_equal(current_step['enemy_y'], self.previous_step['enemy_y']):

                    return 0
        self.previous_step = history_dict
        return 1

# filter_steps = 0
# def this_step(history):
#     global filter_steps
#     if type(filter_steps) == int:
#         filter_steps = history['player_relative_screen']
#         return 1
#     else:
#         current_step = history['player_relative_screen']
#         if np.array_equal(current_step['neutral_x'], filter_steps['neutral_x']) and \
#                 np.array_equal(current_step['neutral_y'], filter_steps['neutral_y']) and \
#                 np.array_equal(current_step['enemy_x'], filter_steps['enemy_x']) and \
#                 np.array_equal(current_step['enemy_y'], filter_steps['enemy_y']):
#
#             return 0
#     filter_steps = history['player_relative_screen']
#     return 1



def filter_substeps(history):
    '''Returns a history where substeps are filtered out'''
    current_step = 0
    previous_step = 0
    new_history = []
    for i in range(len(history)):
        step = history[i]
        current_step = step
        if not previous_step:
            previous_step = current_step
            new_history.append(current_step)
            continue
        #the assumption that the same beacon postions means the same episode
        if np.array_equal(current_step['neutral_x'], previous_step['neutral_x']) and \
                np.array_equal(current_step['neutral_y'], previous_step['neutral_y']) and \
                np.array_equal(current_step['enemy_x'], previous_step['enemy_x']) and \
                np.array_equal(current_step['enemy_y'], previous_step['enemy_y']):
            continue
        else:
            previous_step = current_step
            new_history.append(current_step)

        previous_step = current_step




    return new_history