import logging
import sys
import os
import shutil
import sys
from datetime import datetime
from functools import partial
import tensorflow as tf
from absl import flags
from actorcritic.agent import ActorCriticAgent, ACMode
from actorcritic.runner import Runner, PPORunParams
from common.multienv import SubprocVecEnv, make_sc2env, SingleEnv
from actup_agent import ActupAgent
import numpy as np

import time

import pickle

#import threading

FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("resolution", 64, "Resolution for screen and minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 1, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 100, "Number of complete episodes")
flags.DEFINE_integer("n_steps_per_batch", None,
    "Number of steps per batch, if None use 8 for a2c and 128 for ppo")  # (MINE) TIMESTEPS HERE!!!
flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
flags.DEFINE_string("model_name", "my_beacon_beta_model", "Name for checkpoints and tensorboard summaries")
flags.DEFINE_integer("K_batches", 3,
    "Number of training batches to run in thousands, use -1 to run forever") #(MINE) not for now
flags.DEFINE_string("map_name", "MoveToBeacon_random_stoch", "Name of a map to use.")
flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
flags.DEFINE_boolean("training", False,
    "if should train the model, if false then save only episode score summaries"
)
flags.DEFINE_enum("if_output_exists", "fail", ["fail", "overwrite", "continue"],
    "What to do if summary and model output exists, only for training, is ignored if notraining")
flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
flags.DEFINE_float("loss_value_weight", 1.0, "good value might depend on the environment")
flags.DEFINE_float("entropy_weight_spatial", 1e-6,
    "entropy of spatial action distribution loss weight")
flags.DEFINE_float("entropy_weight_action", 1e-6, "entropy of action-id distribution loss weight")
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo")
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")
flags.DEFINE_enum("agent_mode", ACMode.A2C, [ACMode.A2C, ACMode.PPO], "if should use A2C or PPO")
flags.DEFINE_float('memory_temperature',1.0,'temperature for pyactup')
flags.DEFINE_float('memory_decay', 0.50, 'decay')
flags.DEFINE_float('memory_noise', 0.25, 'noise')
flags.DEFINE_float('memory_mismatch', 1.0, 'mismatch penalty')
flags.DEFINE_float('memory_threshold', -100.0, 'retrieval threshold')
#flags.DEFINE_integer('n_participants', 10, 'number of act-r agents')
flags.DEFINE_list('id_start_end', [181,200], 'id start and end - also defines the number of participants.')

FLAGS(sys.argv)


def main():
    env_args = dict(
        map_name=FLAGS.map_name,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        screen_size_px=(FLAGS.resolution,) * 2,
        minimap_size_px=(FLAGS.resolution,) * 2,
        visualize=FLAGS.visualize,
        replay_dir='/Users/constantinos/Documents/StarcraftMAC/MyAgents/'
    )
    #Create the environment and the agent
    env = SingleEnv(make_sc2env(**env_args))
    data = {'obs_raw': [], 'chunk': [], 'bad_end': [], 'err': [],
            'id': [], 'decision_index': [],'green':[],'orange':[],'blocking':[],'target':[],'value':[],
            'target_0_green_salience':[], 'target_0_orange_salience':[],
            'target_0_target_salience':[], 'target_0_value_estimate':[],
            'target_1_green_salience':[], 'target_1_orange_salience':[],
            'target_1_target_salience':[], 'target_1_value_estimate':[],
            'action_salience_green':[], 'action_salience_orange':[], 'action_salience_target':[],
            'action_salience_blocking':[], 'action_salience_action_estimate':[],
            'noise':[], 'decay':[]}
    # data_list = []
    for i in range(flags.FLAGS.id_start_end[0],flags.FLAGS.id_start_end[1]+1):

        actupAgent = ActupAgent(env,noise=flags.FLAGS.memory_noise,decay=flags.FLAGS.memory_decay)
        obs_raw = actupAgent.reset()
        bad_end = False


        loop_count = 0
        while actupAgent.episode_counter <= (FLAGS.episodes - 1):
            err = ''
            if obs_raw[0].last():
                actupAgent.cumulative_score = 0
                noop = actupAgent.action_processer.process(actupAgent.noop,
                                                     np.reshape(np.asarray([0, 0], dtype=int), (1, 2)))
                obs_raw = env.step(noop)
                print('shiit')
            # time.sleep(0.01)



            data['obs_raw'].append(obs_raw)
            chunk, obs_raw, bad_end, err, value_saliences, action_salience  = actupAgent.decision(obs_raw)
            if bad_end:
                data['obs_raw'] = data['obs_raw'][:-1] #remove the last obs raw because episode won't be recorded
                actupAgent.episode_counter -= 1
                continue #i.e. don't record this episode, act like it didn't exist.
            #print(chunk)
            data['chunk'].append(chunk)
            data['bad_end'].append(bad_end)
            data['err'].append(err)
            data['id'].append(i)
            data['decision_index'].append(loop_count)
            data['green'].append(chunk['green'])
            data['orange'].append(chunk['orange'])
            data['blocking'].append(chunk['blocking'])
            data['target'].append(chunk['target'])
            data['value'].append(chunk['value'])
            data['noise'].append(flags.FLAGS.memory_noise)
            data['decay'].append(flags.FLAGS.memory_decay)
            for sal in value_saliences:
                if sal['imagined_target'] == 0:
                    #imagining the orange is the target, what is the value estimate?
                    data['target_0_orange_salience'].append(sal['orange'])
                    data['target_0_green_salience'].append(sal['green'])
                    data['target_0_target_salience'].append(sal['target'])
                    data['target_0_value_estimate'].append(sal['value_estimate'])
                    # data['target_1_orange_salience'].append(np.NaN)
                    # data['target_1_green_salience'].append(np.NaN)
                    # data['target_1_target_salience'].append(np.NaN)
                    # data['target_1_value_estimate'].append(np.NaN)
                else:
                    data['target_1_orange_salience'].append(sal['orange'])
                    data['target_1_green_salience'].append(sal['green'])
                    data['target_1_target_salience'].append(sal['target'])
                    data['target_1_value_estimate'].append(sal['value_estimate'])
                    # data['target_0_orange_salience'].append(np.NaN)
                    # data['target_0_green_salience'].append(np.NaN)
                    # data['target_0_target_salience'].append(np.NaN)
                    # data['target_0_value_estimate'].append(np.NaN)
            if not value_saliences:
                data['target_1_orange_salience'].append(np.NaN)
                data['target_1_green_salience'].append(np.NaN)
                data['target_1_target_salience'].append(np.NaN)
                data['target_1_value_estimate'].append(np.NaN)
                data['target_0_orange_salience'].append(np.NaN)
                data['target_0_green_salience'].append(np.NaN)
                data['target_0_target_salience'].append(np.NaN)
                data['target_0_value_estimate'].append(np.NaN)

            data['action_salience_orange'].append(action_salience['orange'])
            data['action_salience_green'].append(action_salience['green'])
            data['action_salience_target'].append(action_salience['target'])
            data['action_salience_blocking'].append(action_salience['blocking'])
            data['action_salience_action_estimate'].append(action_salience['action'])
            loop_count += 1
            actupAgent.memory.learn(**chunk)

            # if bad_end:
            #     bad_end = False
            #     data['obs_raw'].append(obs_raw)
            #     chunk, obs_raw, bad_end, err = actupAgent.decision(obs_raw)
            #     data['chunk'].append(chunk)
            #     data['bad_end'].append(bad_end)
            #     data['err'].append(err)
            #     if bad_end:
            #         actupAgent.episode_counter -= 1
            #         obs_raw = actupAgent.reset()
            #         print('bad end')
            # if not bad_end:
            #     print("success")
            #     actupAgent.memory.learn(**chunk)
            # else:
            #     print('ended wrong')
            #     bad_end = False
        # data_list.append(data.copy())
    pickle.dump(data, open('data_' + repr(flags.FLAGS.id_start_end[0]) + '_' + repr(flags.FLAGS.id_start_end[1]) + '.pkl','wb'))

if __name__ == "__main__":
    main()

