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
from gym_FPS.utils import *
import Memory

buffer_size = 10000
resolution = (100,100)
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

def get_action(state, flag, env, dqn):
    action = []
    s = []
    command_size = 14



    if flag == 'myself':
        for i in range(len(env.units_id)):
            uid = env.units_id[i];
            ut = env.myunits[uid]
            if ut['HEALTH'] > 0:
                a = dqn.choose_action(state[i], command_size, epsilon=0, enjoy=True)
                action.append(a)
            else:
                action.append(-1) # -1 means invalid action
        
    else:
        for i in range(len(env.units_e_id)):
            uid = env.units_e_id[i]
            ut = env.myunits[uid]
            if ut['HEALTH'] > 0 :
                ut = env.myunits[uid]
                target = get_weakest(ut['POSITION'][0], ut['POSITION'][2],env.state['units_myself'])
                a = -1
                for i in range(len(env.units_id)):
                    my_uid = env.units_id[i]
                    if my_uid is target:
                        a = 9 + i

                action.append(a) 
            else:
                action.append(-1)

    return action

def Micro_Management(episodes, env, dqn):
    env.obs = env._make_observation()
    unit_size = len(env.state['units_myself'])
    screen_my = env.obs
    current_step = 0
    done = False
    rewards = []
    epi_flag = True
    tmp_loss = 0
    reward = 0

    cumulative_reward = 0
    unatural_flag = False

    while not done:
        if current_step >= 200:
            epi_flag = False
            break
        else:
            current_step += 1

        print("before get_action")
        action= get_action(screen_my, 'myself', env, dqn)
        print("action:{}".format(action))
            
        print("after get_action")

        screen_my_n, reward, done, unit_size_ = env.step(action)#执行完动作后的时间，time2
        print("after step actions")

        if reward is not None:
            rewards.append(reward)

        screen_my = screen_my_n

        current_step += 1
        if reward is not None:
            cumulative_reward += reward
    if epi_flag:
        episodes += 1
        if bool(env.state['battle_won']):
            battles_won += 1

        wf(str(battles_won) + '\n' + str(episodes), 0)
        wf(str(np.mean(rewards)) + '\n' + str(episodes), 1)
        print('episodes:', episodes, ', win:', battles_won, )
        if episodes % CONFIG.episode_to_reset_win == 0:
            win_rate[episodes] = battles_won / CONFIG.episode_to_reset_win
            print('win rate:', win_rate[episodes])
            battles_won = 0

    return epi_flag