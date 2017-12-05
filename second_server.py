# Part of this are from easy_scripted_test in pysc2 folder
import sys
from absl import flags
import second_agent
from pysc2.env import run_loop
from pysc2.env import sc2_env

import json

from twisted.internet import protocol, reactor
from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Protocol, DatagramProtocol, ClientFactory

import time
import threading


FLAGS = flags.FLAGS

#from pysc2.agents import scripted_agent
#from pysc2.lib import stopwatch
#from pysc2.tests import utils
#from absl.testing import absltest as basetest

# see at the end for utilis and TestCase (delete this comment if you use it for replays
#class TestEasy(utils.TestCase):




steps = 400 # = 4 Episodes
step_mul = 16

class MasterServerProtocol(LineReceiver):
    delimiter = b'\n'

    def connectionMade(self):
        print("Connection Made.")

    def lineReceived(self, line):
        #print("Received line", line)
        jLine = None
        try:
            jLine = json.loads(line)
        except ValueError:
            print
            'Line', line, 'is not JSON'
            return 0

        if 'command' in jLine:
            if jLine['command'] == 'tic':
                self.factory.agent.tickable = True
        if 'get_observation' in jLine:
            r_dict = self.factory.agent.get_observation(None)
            print( "sending", bytes(json.dumps(r_dict), 'UTF-8'))
            self.sendLine(bytes(json.dumps(r_dict), 'UTF-8'))
            self.sendLine(bytes(json.dumps({}), 'UTF-8')) #neded on the lisp side to flush...
            #print("r_dict", r_dict)
            #minor change


        return 0

server_agent = second_agent.MoveToBeacon()


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




if __name__ == '__main__':

    thread = threading.Thread(target=game_thread, args=sys.argv)
    thread.start()
    print("here")
    #time.sleep(30)
    factory = protocol.ServerFactory()
    factory.protocol = MasterServerProtocol
    factory.agent = server_agent

    reactor.listenTCP(33333, factory)

    reactor.run()









# def main():
#     FLAGS(sys.argv)
#     with sc2_env.SC2Env(
#         map_name="MoveToBeacon",
#         step_mul=step_mul,
#         game_steps_per_episode=steps * step_mul,
#         save_replay_episodes=5,
#         #replay_dir='/Users/paulsomers/StarcraftMAC/MyAgents/',
#         screen_size_px=(128, 128)) as env:
#         # agent = scripted_agent.MoveToBeacon()
#         myAgent = agent.MoveToBeacon()
#         run_loop.run_loop([myAgent], env, steps)
#         print('Reward: ', myAgent.reward, ' | Episodes: ', myAgent.episodes)
#         #env.save_replay('/Users/paulsomers/StarcraftMAC/MyAgents/')

    # Get some points
#    self.assertLessEqual(agent.episodes, agent.reward)
#    self.assertEqual(agent.steps, self.steps)

  # def test_collect_mineral_shards(self):
  #   with sc2_env.SC2Env(
  #       map_name="CollectMineralShards",
  #       step_mul=self.step_mul,
  #       game_steps_per_episode=self.steps * self.step_mul) as env:
  #     agent = scripted_agent.CollectMineralShards()
  #     run_loop.run_loop([agent], env, self.steps)
  #
  #   # Get some points
  #   self.assertLessEqual(agent.episodes, agent.reward)
  #   self.assertEqual(agent.steps, self.steps)
  #
  # def test_defeat_roaches(self):
  #   with sc2_env.SC2Env(
  #       map_name="DefeatRoaches",
  #       step_mul=self.step_mul,
  #       game_steps_per_episode=self.steps * self.step_mul) as env:
  #     agent = scripted_agent.DefeatRoaches()
  #     run_loop.run_loop([agent], env, self.steps)
  #
  #   # Get some points
  #   self.assertLessEqual(agent.episodes, agent.reward)
  #   self.assertEqual(agent.steps, self.steps)


# if __name__ == "__main__":
#     #basetest.main()
#     main()
