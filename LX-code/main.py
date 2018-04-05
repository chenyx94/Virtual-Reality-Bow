#coding=utf8

import numpy as np
import gym_FPS
from gym_FPS.envs.starcraft.Config import Config
import argparse
import pickle
from gym_FPS.envs.starcraft.model import DDPG, DQN, DQN_normal
import tensorflow as tf
import pylab
import time
import os, gym
from control import Attack, Dispatch, Priority
from gym_FPS.utils import *
from MM import *

# action:(attack or move, degree, distance)
# state:()
#  * hit points, cooldown, ground range, is enemy, degree, distance (myself)
#  * hit points, cooldown, ground range, is enemy (enemy)
CONFIG = Config()
MEMORY_CAPACITY = CONFIG.memory_capacity
parser = argparse.ArgumentParser()
parser.add_argument('--ip', help='server ip', default=CONFIG.serverip)
parser.add_argument('--port', help='server port', default=CONFIG.serverport)
parser.add_argument('--result', help='result', default='result_second')
args = parser.parse_args()
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.result + '/model'):
    os.mkdir(args.result + '/model')
if not os.path.exists(args.result + '/model_e'):
    os.mkdir(args.result + '/model_e')

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_round = 10
batch_size = 16

def wf(str, flag):
    if flag == 0:
        filepath = args.result + '/win.txt'
    else:
        filepath = args.result + '/reward.txt'
    F_battle = open(filepath, 'a')
    F_battle.write(str + '\n')
    F_battle.close()

if __name__ == '__main__':

    #----------------------------------init env------------------------------------------
    env = gym.make('FPSSingle-v0')
    print("begin init env....")
    env.set_env(args.ip, 5123, client_DEBUG=False, env_DEBUG=False, speedup=CONFIG.speed)
    env.restart(port=args.port)
    env.seed(123)
    sess = tf.Session()
    print("finish init env!")
    #----------------------------------init memory--------------------------------------------------
    s_dim = resolution
    a_dim = 14 
    #----------------------------------init network------------------------------------------------
    print("begin init network.....")

    if CONFIG.load == 1:
        print("OLD VAR LOADED")
        ckpt = tf.train.get_checkpoint_state(args.result + '/model')
        if ckpt and ckpt.model_checkpoint_path:
            print("OLD VARS!")
            dqn.saver.restore(sess, ckpt.model_checkpoint_path)
    print("finish init network")


    episodes = CONFIG.episode
    battles_won = 0
    attack = Attack()
    dispatcher = Dispatch()
    priority = Priority()


    var = 0.3 * (0.9999 ** CONFIG.episode)  # control exploration
    #print('var is ', var)
    win_rate = {}
    while episodes < 10000:
        s, unit_size, e_unit_size = env.reset()
        screen_my = s
        current_step = 0
        done = False
        rewards = []
        epi_flag = True
        tmp_loss = 0
        reward = None

        cumulative_reward = 0
        unatural_flag = False
        print("finish reset env!")

        while not attack.act(s_):

            """
            #formation algorithm

			rank = form.formation(s)

            #policy of rank
			action = policy(rank)
			s  = env.policy_step(action)

            #Decide whether to fight
            while support > 0:
                s_, support = get_state()
                if s_ != s:
                    done = Done.is_done(s_)

                if done:
                    break
                s = s_
            if done:
                break
            """

        epi_flag = micro_management.Micro_Management(episodes, env, dqn)
        """
        if epi_flag:
            #### compute reward ####
            R = env.compute_reward()
            form.formation_store(R)
        form.learn()
        Done.learn()
        """
