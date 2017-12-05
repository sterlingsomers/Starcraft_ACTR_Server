import threading
import time

import sys
from absl import flags
import sc2_agent
from pysc2.env import run_loop
from pysc2.env import sc2_env


import actr

actr.load_act_r_model("/Users/paulsomers/StarcraftMAC/MyAgents/starcraft-A1rev3c.lisp")

#game setup
FLAGS = flags.FLAGS
steps = 400 # = 4 Episodes
step_mul = 16

server_agent = sc2_agent.MoveToBeacon()
server_agent.actr_setup(actr)

def game_thread(*args):
    FLAGS(args)
    with sc2_env.SC2Env(
        map_name="MoveToBeacon",
        #map_name="MoveToBeacon",
        step_mul=step_mul,
        game_steps_per_episode=steps * step_mul,
        save_replay_episodes=5,
        replay_dir='/Users/paulsomers/StarcraftMAC/MyAgents/',
        screen_size_px=(128, 128)) as env:
    #         # agent = scripted_agent.MoveToBeacon()
            myAgent = server_agent
            run_loop.run_loop([myAgent], env, steps)
            print('Reward: ', myAgent.reward, ' | Episodes: ', myAgent.episodes)
            #env.save_replay('/Users/paulsomers/StarcraftMAC/MyAgents/')


#start the game
if __name__ == '__main__':

    thread = threading.Thread(target=game_thread, args=sys.argv)
    thread.start()

    time.sleep(20)

    act_thread = threading.Thread(target=actr.run, args=[1000])
    act_thread.start()
    #for x in range(1000):
    #   actr.process_events()

    print(actr.mp_models())
    print("test")


