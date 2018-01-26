import collections
import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.optimizers import OPTIMIZER_SUMMARIES
from actorcritic.policy import FullyConvPolicy
from common.preprocess import ObsProcesser, FEATURE_KEYS, AgentInputTuple
from common.util import weighted_random_sample, select_from_each_row, ravel_index_pairs


#TODO why doesn't it already import the features?
from pysc2.lib import features

import actr
import math
import time
import threading

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

def _get_placeholders(spatial_dim):
    sd = spatial_dim
    feature_list = [
        (FEATURE_KEYS.minimap_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_MINIMAP_CHANNELS]),
        (FEATURE_KEYS.screen_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_SCREEN_CHANNELS]),
        (FEATURE_KEYS.screen_unit_type, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.is_spatial_action_available, tf.float32, [None]),
        (FEATURE_KEYS.available_action_ids, tf.float32, [None, len(actions.FUNCTIONS)]),
        (FEATURE_KEYS.selected_spatial_action, tf.int32, [None, 2]),
        (FEATURE_KEYS.selected_action_id, tf.int32, [None]),
        (FEATURE_KEYS.value_target, tf.float32, [None]),
        (FEATURE_KEYS.player_relative_screen, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.player_relative_minimap, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.advantage, tf.float32, [None])
    ]
    return AgentInputTuple(
        **{name: tf.placeholder(dtype, shape, name) for name, dtype, shape in feature_list}
    )


class ACMode:
    A2C = "a2c"
    PPO = "ppo"


SelectedLogProbs = collections.namedtuple("SelectedLogProbs", ["action_id", "spatial", "total"])


class ActorCriticAgent:
    _scalar_summary_key = "scalar_summaries"

    def __init__(self,
            sess: tf.Session,
            summary_path: str,
            all_summary_freq: int,
            scalar_summary_freq: int,
            spatial_dim: int,
            mode: str,
            clip_epsilon=0.2,
            unit_type_emb_dim=4,
            loss_value_weight=1.0,
            entropy_weight_spatial=1e-6,
            entropy_weight_action_id=1e-5,
            max_gradient_norm=None,
            optimiser="adam",
            optimiser_pars: dict = None,
            policy=FullyConvPolicy
    ):
        """
        Actor-Critic Agent for learning pysc2-minigames
        https://arxiv.org/pdf/1708.04782.pdf
        https://github.com/deepmind/pysc2

        Can use
        - A2C https://blog.openai.com/baselines-acktr-a2c/ (synchronous version of A3C)
        or
        - PPO https://arxiv.org/pdf/1707.06347.pdf

        :param summary_path: tensorflow summaries will be created here
        :param all_summary_freq: how often save all summaries
        :param scalar_summary_freq: int, how often save scalar summaries
        :param spatial_dim: dimension for both minimap and screen
        :param mode: a2c or ppo
        :param clip_epsilon: epsilon for clipping the ratio in PPO (no effect in A2C)
        :param loss_value_weight: value weight for a2c update
        :param entropy_weight_spatial: spatial entropy weight for a2c update
        :param entropy_weight_action_id: action selection entropy weight for a2c update
        :param max_gradient_norm: global max norm for gradients, if None then not limited
        :param optimiser: see valid choices below
        :param optimiser_pars: optional parameters to pass in optimiser
        :param policy: Policy class
        """

        assert optimiser in ["adam", "rmsprop"]
        assert mode in [ACMode.A2C, ACMode.PPO]
        self.mode = mode
        self.sess = sess
        self.spatial_dim = spatial_dim
        self.loss_value_weight = loss_value_weight
        self.entropy_weight_spatial = entropy_weight_spatial
        self.entropy_weight_action_id = entropy_weight_action_id
        self.unit_type_emb_dim = unit_type_emb_dim
        self.summary_path = summary_path
        os.makedirs(summary_path, exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(summary_path)
        self.all_summary_freq = all_summary_freq
        self.scalar_summary_freq = scalar_summary_freq
        self.train_step = 0
        self.max_gradient_norm = max_gradient_norm
        self.clip_epsilon = clip_epsilon
        self.policy = policy

        #load the ACT-R model
        self.actr = actr
        self.actr.load_act_r_model("/Users/paulsomers/StarcraftMAC/MyAgents/starcraft-B1-rev2.lisp")
        self.actr.add_command("tic", self.do_tic)
        self.actr.add_command("ignore", self.ignore)
        self.actr.add_command("set_response", self.set_response)
        self.actr.add_command("RHSWait", self.RHSWait)
        self.actr.add_command("GameStartWait", self.game_start_wait)

        self.actr.add_command("consine_similarity", self.cosine_similarity)


        #some act-r items
        self.tickable = False
        self.game_start_wait_flag = True
        self.stepper_waiting = False
        self.stepper_started = False
        self.stepped = False
        self.obs = None
        self.actrChunks = []




        opt_class = tf.train.AdamOptimizer if optimiser == "adam" else tf.train.RMSPropOptimizer
        if optimiser_pars is None:
            pars = {
                "adam": {
                    "learning_rate": 1e-4,
                    "epsilon": 5e-7
                },
                "rmsprop": {
                    "learning_rate": 2e-4
                }
            }[optimiser]
        else:
            pars = optimiser_pars
        self.optimiser = opt_class(**pars)

    def init(self):
        self.sess.run(self.init_op)
        if self.mode == ACMode.PPO:
            self.update_theta()

    def _get_select_action_probs(self, pi, selected_spatial_action_flat):
        action_id = select_from_each_row(
            pi.action_id_log_probs, self.placeholders.selected_action_id
        )
        spatial = select_from_each_row(
            pi.spatial_action_log_probs, selected_spatial_action_flat
        )
        total = spatial + action_id

        return SelectedLogProbs(action_id, spatial, total)

    def _scalar_summary(self, name, tensor):
        tf.summary.scalar(name, tensor,
            collections=[tf.GraphKeys.SUMMARIES, self._scalar_summary_key])

    def cosine_similarity(self,arg1,arg2):
        pass

    def is_blocking(self,seg1,seg2,point):
        '''if green=point is between player=seg1 and orange=seg2'''
        #https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment
        crossproduct = (point[1] - seg1[1]) * (seg2[0] - seg1[0]) - (point[0] - seg1[0]) * (seg2[1] - seg1[1])
        if abs(crossproduct) != 0:
            return False

        dotproduct = (point[0] - seg1[0]) * (seg2[0] - seg1[0]) + (point[1] - seg1[1]) * (seg2[1] - seg1[1])
        if dotproduct < 0:
            return False

        squaredlengthba = (seg2[0] - seg1[0]) * (seg2[0] - seg1[0]) + (seg2[1] - seg1[1]) * (seg2[1] - seg1[1])
        if dotproduct > squaredlengthba: return False

        return True

    def distance(self,a,b):
        '''duh. distance betweeen a and b'''
        pass

    def RHSWait(self):
        print("RHSWwait called, flag set to True")
        self.RHSWaitFlag = True
        while self.RHSWaitFlag:
            time.sleep(0.00001)
        return 1

    def ignore(self):
        return 0

    def game_start_wait(self):
        print("GAME START WAIT")

        while self.game_start_wait_flag:
            time.sleep(0.00001)
        print("Game START returned")
        return 1

    def do_tic(self):
        print("do_tic: tic called")
        #print("do_tic: waiting for the stepper to start")
        self.actr.schedule_simple_event_now("ignore")
        self.tickable = True

        return 1

    def push_observation(self, args):
        '''Return a dictionary of observations'''
        player_relative = self.obs["player_relative_screen"]
        #print("PR", (player_relative == _PLAYER_NEUTRAL).nonzero())
        orange_beacon = False
        green_beacon = False
        player = False
        between = False

        neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()[1:3]
        enemy_x, enemy_y = (player_relative == _PLAYER_HOSTILE).nonzero()[1:3]
        player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()[1:3]

        if neutral_y.any():
            orange_beacon = True

            if enemy_y.any():
                orange_beacon = True
            if player_y.any():
                player = True

        if orange_beacon and player and green_beacon:
            #check for blocking or overlap

            #Determine if green is between orange and player
            green_points = zip(neutral_x,neutral_y)
            orange_points = zip(enemy_x, enemy_y)
            player_points = zip(player_x, player_y)

            #https: // stackoverflow.com / questions / 328107 / how - can - you - determine - a - point - is -between - two - other - points - on - a - line - segment
            between = False
            set_of_points_to_check_between = [(x,y) for x in player_points for y in orange_points]
            for points in set_of_points_to_check_between:
                for green_points in player_points:
                    if self.is_blocking(points[0],points[1],player_points):
                        between = True
            print("BETWEEN", between)


            # print(neutral_y, len(neutral_y), min(neutral_y), max(neutral_y))


            chk = self.actr.define_chunks(['neutral_x', int(neutral_x.mean()), 'neutral_y', int(neutral_y.mean()),'wait', 'false'])
            # the wait, false is for to make sure something other than the wait production fires.

            #not sure I need the actrChunks list
            #self.actrChunks.append(chk)

            #self.actr.schedule_simple_event_now("ignore")
            #self.actr.set_buffer_chunk('imaginal', chk[0])
            self.actr.schedule_simple_event_now("set-buffer-chunk", ['imaginal', chk[0]])#self.actr.set_buffer_chunk('imaginal',chk[0])
            #self.actr.schedule_simple_event_now("ignore")

        return 1
    def set_response(self,*args):
        print("set_response:", args)
        args = list(args)



        if args[0] == "_SELECT_ARMY":
            self.response = [_SELECT_ARMY, [_SELECT_ALL]]
        elif args[0] == "_MOVE_SCREEN":
            self.response = [_MOVE_SCREEN, [_NOT_QUEUED, [args[2][1],args[3][1]]]]
        else:
            pass


        print("RES:", self.response)

        self.do_tic()

        return 1

    def build_model(self):
        self.placeholders = _get_placeholders(self.spatial_dim)

        with tf.variable_scope("theta"):
            theta = self.policy(self, trainable=True).build()

        selected_spatial_action_flat = ravel_index_pairs(
            self.placeholders.selected_spatial_action, self.spatial_dim
        )

        selected_log_probs = self._get_select_action_probs(theta, selected_spatial_action_flat)

        # maximum is to avoid 0 / 0 because this is used to calculate some means
        sum_spatial_action_available = tf.maximum(
            1e-10, tf.reduce_sum(self.placeholders.is_spatial_action_available)
        )

        neg_entropy_spatial = tf.reduce_sum(
            theta.spatial_action_probs * theta.spatial_action_log_probs
        ) / sum_spatial_action_available
        neg_entropy_action_id = tf.reduce_mean(tf.reduce_sum(
            theta.action_id_probs * theta.action_id_log_probs, axis=1
        ))

        if self.mode == ACMode.PPO:
            # could also use stop_gradient and forget about the trainable
            with tf.variable_scope("theta_old"):
                theta_old = self.policy(self, trainable=False).build()

            new_theta_var = tf.global_variables("theta/")
            old_theta_var = tf.global_variables("theta_old/")

            assert len(tf.trainable_variables("theta/")) == len(new_theta_var)
            assert not tf.trainable_variables("theta_old/")
            assert len(old_theta_var) == len(new_theta_var)

            self.update_theta_op = [
                tf.assign(t_old, t_new) for t_new, t_old in zip(new_theta_var, old_theta_var)
            ]

            selected_log_probs_old = self._get_select_action_probs(
                theta_old, selected_spatial_action_flat
            )
            ratio = tf.exp(selected_log_probs.total - selected_log_probs_old.total)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            )
            l_clip = tf.minimum(
                ratio * self.placeholders.advantage,
                clipped_ratio * self.placeholders.advantage
            )
            self.sampled_action_id = weighted_random_sample(theta_old.action_id_probs)
            self.sampled_spatial_action = weighted_random_sample(theta_old.spatial_action_probs)
            self.value_estimate = theta_old.value_estimate
            self._scalar_summary("action/ratio", tf.reduce_mean(clipped_ratio))
            self._scalar_summary("action/ratio_is_clipped",
                tf.reduce_mean(tf.to_float(tf.equal(ratio, clipped_ratio))))
            policy_loss = -tf.reduce_mean(l_clip)
        else:
            self.sampled_action_id = weighted_random_sample(theta.action_id_probs)
            self.sampled_spatial_action = weighted_random_sample(theta.spatial_action_probs)
            self.value_estimate = theta.value_estimate
            policy_loss = -tf.reduce_mean(selected_log_probs.total * self.placeholders.advantage)

        value_loss = tf.losses.mean_squared_error(
            self.placeholders.value_target, theta.value_estimate)

        loss = (
            policy_loss
            + value_loss * self.loss_value_weight
            + neg_entropy_spatial * self.entropy_weight_spatial
            + neg_entropy_action_id * self.entropy_weight_action_id
        )

        self.train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.optimiser,
            clip_gradients=self.max_gradient_norm,
            summaries=OPTIMIZER_SUMMARIES,
            learning_rate=None,
            name="train_op"
        )

        self._scalar_summary("value/estimate", tf.reduce_mean(self.value_estimate))
        self._scalar_summary("value/target", tf.reduce_mean(self.placeholders.value_target))
        self._scalar_summary("action/is_spatial_action_available",
            tf.reduce_mean(self.placeholders.is_spatial_action_available))
        self._scalar_summary("action/selected_id_log_prob",
            tf.reduce_mean(selected_log_probs.action_id))
        self._scalar_summary("loss/policy", policy_loss)
        self._scalar_summary("loss/value", value_loss)
        self._scalar_summary("loss/neg_entropy_spatial", neg_entropy_spatial)
        self._scalar_summary("loss/neg_entropy_action_id", neg_entropy_action_id)
        self._scalar_summary("loss/total", loss)
        self._scalar_summary("value/advantage", tf.reduce_mean(self.placeholders.advantage))
        self._scalar_summary("action/selected_total_log_prob",
            tf.reduce_mean(selected_log_probs.total))
        self._scalar_summary("action/selected_spatial_log_prob",
            tf.reduce_sum(selected_log_probs.spatial) / sum_spatial_action_available)

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.all_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
        self.scalar_summary_op = tf.summary.merge(tf.get_collection(self._scalar_summary_key))

    def _input_to_feed_dict(self, input_dict):
        return {k + ":0": v for k, v in input_dict.items()}

    def step(self, obs):
        feed_dict = self._input_to_feed_dict(obs)

        #start ACT-R (after the game has started)
        if self.game_start_wait_flag:
            chk = self.actr.define_chunks(
                ['wait', 'false'])

            self.actr.schedule_simple_event_now("set-buffer-chunk",
                                                ['imaginal', chk[0]])  # self.actr.set_buffer_chunk('imaginal',chk[0])
            actrThread = threading.Thread(target=self.actr.run, args=[100])
            actrThread.start()
            self.game_start_wait_flag = False

        #for key in obs:
        #    print("KEY", key)
        #insert some ACTR things

        print("here")
        # this is a temporary solution for resetting...

        #TODO obs["available_action_ids"] is incorrect, I think.
        #TODO what it SEEMS to be is an array of binary values. 1 = action available, 0 = not available.

        if not obs["available_action_ids"][0][_MOVE_SCREEN]:
            current_goal_chunk = self.actr.buffer_chunk('goal')
            self.actr.mod_chunk(current_goal_chunk[0], "state", "select-army")


        print("here2")
        self.obs = obs
        w = self.push_observation(None)
        #current_imaginal_chunk = self.actr.buffer_chunk('imaginal')
        #print("current_imaginal_chunk", current_imaginal_chunk[0])
        #self.actr.mod_chunk(current_imaginal_chunk[0], "wait", "false")
        self.RHSWaitFlag = False
        print("RHSWaitFlag set to False")
        # self.actr.schedule_simple_event_now("mod-chunk-fct", 'imaginal', 'wait', 'false')

        print("here3")
        while not self.tickable:
            time.sleep(0.00001)

        self.stepped = True
        self.tickable = False

        print("here4")

        action_id, spatial_action, value_estimate = self.sess.run(
            [self.sampled_action_id, self.sampled_spatial_action, self.value_estimate],
            feed_dict=feed_dict
        )

        spatial_action_2d = np.array(
            np.unravel_index(spatial_action, (self.spatial_dim,) * 2)
        ).transpose()

        return action_id, spatial_action_2d, value_estimate

    def train(self, input_dict):
        feed_dict = self._input_to_feed_dict(input_dict)
        ops = [self.train_op]

        write_all_summaries = (
            (self.train_step % self.all_summary_freq == 0) and
            self.summary_path is not None
        )
        write_scalar_summaries = (
            (self.train_step % self.scalar_summary_freq == 0) and
            self.summary_path is not None
        )

        if write_all_summaries:
            ops.append(self.all_summary_op)
        elif write_scalar_summaries:
            ops.append(self.scalar_summary_op)

        r = self.sess.run(ops, feed_dict)

        if write_all_summaries or write_scalar_summaries:
            self.summary_writer.add_summary(r[-1], global_step=self.train_step)

        self.train_step += 1

    def get_value(self, obs):
        feed_dict = self._input_to_feed_dict(obs)
        return self.sess.run(self.value_estimate, feed_dict=feed_dict)

    def flush_summaries(self):
        self.summary_writer.flush()

    def save(self, path, step=None):
        os.makedirs(path, exist_ok=True)
        step = step or self.train_step
        print("saving model to %s, step %d" % (path, step))
        self.saver.save(self.sess, path + '/model.ckpt', global_step=step)

    def load(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded old model with train_step %d" % self.train_step)
        self.train_step += 1

    def update_theta(self):
        if self.mode == ACMode.PPO:
            self.sess.run(self.update_theta_op)
