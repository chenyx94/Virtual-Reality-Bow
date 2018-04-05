#coding=utf8

import numpy as np
import gym_FPS
from gym_FPS.envs.starcraft.Config import Config
import argparse
import pickle
from gym_FPS.envs.starcraft.model import DDPG
import tensorflow as tf
import pylab
import time
import os, gym
from gym_FPS.utils import *

# action:(attack or move, degree, distance)
# state:()
#  * hit points, cooldown, ground range, is enemy, degree, distance (myself)
#  * hit points, cooldown, ground range, is enemy (enemy)
CONFIG = Config()
MEMORY_CAPACITY = CONFIG.memory_capacity
parser = argparse.ArgumentParser()
parser.add_argument('--ip', help='server ip', default=CONFIG.serverip)
parser.add_argument('--port', help='server port', default=CONFIG.serverport)
parser.add_argument('--result', help='result', default='result')
args = parser.parse_args()
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.result + '/model'):
    os.mkdir(args.result + '/model')
if not os.path.exists(args.result + '/model_e'):
    os.mkdir(args.result + '/model_e')

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def wf(str, flag):
    if flag == 0:
        filepath = args.result + '/win.txt'
    else:
        filepath = args.result + '/reward.txt'
    F_battle = open(filepath, 'a')
    F_battle.write(str + '\n')
    F_battle.close()


def get_next_feature(state, unit_size, flag):
    s = []
    if type(state) is list:
        pass
    elif state.size != 0:
        s_obs = np.asarray(state)
        next_s = s_obs.reshape([-1, s_dim])
        if flag == 'myself':
            for i in range(unit_size):
                s.append(state[i])
        else:
            for i in range(unit_size):
                s.insert(0, state[-1 - i])

        s = np.asarray(s)
        s = s.reshape([-1, s_dim])
        return next_s, s
    return [], []


def get_action(state, unit_size, flag):
    action = []
    s = []
    if type(state) is list:
        pass
    elif state.size != 0:
        try:
            obs_s = np.asarray(state)
            input_s = obs_s.reshape([-1, s_dim])
            if flag == 'myself':
                for i in range(unit_size):
                    s.append(state[i])
            else:
                for i in range(unit_size):
                    s.insert(0, state[-1-i])

            s = np.asarray(s)
            s = s.reshape([-1, s_dim])
            if flag == 'myself':
                action = ddpg.choose_action(input_s, s, unit_size)#这一步有exception
            else:
                action = ddpg_e.choose_action(input_s, s, unit_size)
            action = np.clip(action + np.random.normal(0, var, action.shape), -1, 1)
        except Exception as e:
            print(e)
        return action, input_s, s
    return [], [], []



def store(state, s1, state_, s1_, action, total_reward, unit_size, unit_size_, flag):
    if type(state) is list:
        return
    if type(state_) is list:
        return
    if total_reward is not None and total_reward != 0:
        try:
            if state.shape[0] == state_.shape[0]:
                if flag == 'myself':
                    ddpg.store_transition(state, s1, action, total_reward, state_, s1_, unit_size, unit_size_)
                else:
                    ddpg_e.store_transition(state, s1, action, total_reward, state_, s1_, unit_size, unit_size_)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    env = gym.make('FPSDouble-v0')
    env.set_env(args.ip, 5123, client_DEBUG=False, env_DEBUG=False, speedup=CONFIG.speed)
    env.restart(port=args.port)
    env.seed(123)
    sess = tf.Session()

    s_dim = env.observation_space.shape[1]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(sess, a_dim, s_dim, a_bound, 'Actor', 'Critic')
    ddpg_e = DDPG(sess, a_dim, s_dim, a_bound, 'Actor_e', 'Critic_e')
    if CONFIG.load == 1:
        print("OLD VAR LOADED")
        ckpt = tf.train.get_checkpoint_state(args.result + '/model')
        ckpt_e = tf.train.get_checkpoint_state(args.result + '/model_e')
        if ckpt and ckpt.model_checkpoint_path:
            print("OLD VARS!")
            ddpg.saver.restore(sess, ckpt.model_checkpoint_path)
            ddpg_e.saver.restore(sess, ckpt_e.model_checkpoint_path)



    episodes = CONFIG.episode
    battles_won = 0

    var = 0.3 * (0.9999 ** CONFIG.episode)  # control exploration
    #print('var is ', var)
    win_rate = {}
    while True:
        s, unit_size, e_unit_size = env.reset()
        current_step = 0
        done = False
        rewards = []
        epi_flag = True
        tmp_loss = 0
        reward = 0

        cumulative_reward = 0
        unatural_flag = False
        while not done:
            '''unatural = maxmin_distance(env.state['units_myself'], env.state['units_enemy'])
            if unatural or unatural_flag and current_step != 0:
#                done = env.die_fast()
                unatural_flag = True
                epi_flag = False
                print("unatural")
                break
            elif current_step == 0:
                print("unatural first step")
                time.sleep(2)
                epi_flag = False
                continue
            if current_step >= 800:
                env.restart()
                print("restart")
                epi_flag = False
                break'''
            action, s, s1 = get_action(s, unit_size, 'myself')
            action_e, s, s1_e = get_action(s, e_unit_size, 'enemy')
            s_, reward, done, unit_size_, e_unit_size_ = env.step([action, action_e])#执行完动作后的时间，time2

            s_, s1_ = get_next_feature(s_, unit_size_, 'myself')
            s_, s1_e_ = get_next_feature(s_, e_unit_size_, 'enemy')
            if reward is not None:
                rewards.append(reward)
                store(state=s, s1=s1, action=action, total_reward=reward, state_=s_, s1_=s1_, unit_size=unit_size, unit_size_=unit_size_, flag='myself')
                store(state=s, s1=s1_e, action=action_e, total_reward=-reward, state_=s_, s1_=s1_e, unit_size=e_unit_size, unit_size_=e_unit_size_, flag='enemy')

            var *= 0.9999
            s = s_
            s1 = s1_
            s1_e = s1_e_
            unit_size = unit_size_
            e_unit_size = e_unit_size_

            current_step += 1
            if reward is not None:
                cumulative_reward += reward
        if epi_flag:
            episodes += 1
            if bool(env.state['battle_won']):
                battles_won += 1
            if ddpg.pointer > CONFIG.replay_start_size:
                ddpg.learn()
                ddpg_e.learn()
            wf(str(battles_won) + '\n' + str(episodes), 0)
            wf(str(np.mean(rewards)) + '\n' + str(episodes), 1)
            print('episodes:', episodes, ', win:', battles_won, ', mean_reward:', np.mean(rewards))
            if episodes % CONFIG.episode_to_reset_win == 0:
                win_rate[episodes] = battles_won / CONFIG.episode_to_reset_win
                print('win rate:', win_rate[episodes])
                battles_won = 0


        #print cumulative_reward
        if episodes % CONFIG.episode_to_save == 0 and episodes != 0:
            ddpg.saver.save(sess, args.result + './model/model.ckpt', global_step=episodes)
            ddpg_e.saver.save(sess, args.result + './model_e/model.ckpt', global_step=episodes)



