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

def get_weakest(X, Y, units_table):
    min_total_dist = 1E30
    chosen_uid = -1
    min_total_hp = 1E30
    for uid, ut in units_table.items():
        if ut is None:
            continue
        tmp_hp = ut['HEALTH']
        tmp_dist = (X - ut['POSITION'][0])*(X - ut['POSITION'][0]) + (Y - ut['POSITION'][2])*(Y - ut['POSITION'][2])
        if tmp_hp < min_total_hp:
            min_total_hp = tmp_hp
            chosen_uid = uid
            min_total_dist = tmp_dist
        elif tmp_hp == min_total_hp:
            if tmp_dist < min_total_dist:
                min_total_dist = tmp_dist
                chosen_uid = uid
    return chosen_uid

def get_action(state, flag, env):
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

            
    # action = np.clip(action + np.random.normal(0, var, action.shape), -1, 1)

    return action




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
    buffer_size = 10000
    resolution = (100,100)
    #----------------------------------init env------------------------------------------
    env = gym.make('FPSDouble-v0')
    print("begin init env....")
    env.set_env(args.ip, 5123, client_DEBUG=False, env_DEBUG=False, speedup=CONFIG.speed)
    env.restart(port=args.port)
    env.seed(123)
    sess = tf.Session()
    print("finish init env!")
    #----------------------------------init memory--------------------------------------------------
    replay_buffer = Memory.ReplayMemory_with_dead_index(capacity=buffer_size, resolution=resolution)
    s_dim = resolution 
    a_dim = 14 
    #----------------------------------init network------------------------------------------------
    print("begin init network.....")
    dqn = DQN(sess, resolution=resolution, command_size = 14,index = '0')

    if CONFIG.load == 1:
        print("OLD VAR LOADED")
        ckpt = tf.train.get_checkpoint_state(args.result + '/model')
        if ckpt and ckpt.model_checkpoint_path:
            print("OLD VARS!")
            dqn.saver.restore(sess, ckpt.model_checkpoint_path)
    print("finish init network")


    episodes = CONFIG.episode
    battles_won = 0

    var = 0.3 * (0.9999 ** CONFIG.episode)  # control exploration
    #print('var is ', var)
    win_rate = {}
    while episodes < 10000:
        s, unit_size, e_unit_size = env.reset()
        screen_my = s['screen_my']
        screen_enemy = s['screen_enemy']
        current_step = 0
        done = False
        rewards = []
        epi_flag = True
        tmp_loss = 0
        reward = 0

        cumulative_reward = 0
        unatural_flag = False
        print("finish reset env!")

        while not done:
            if current_step >= 200:
                env.restart()
                print("restart")
                epi_flag = False
                break
            else:
                current_step += 1


            print("before get_action")
            action= get_action(screen_my, 'myself', env)
            print("action:{}".format(action))
            action_e= get_action(screen_enemy, 'enemy', env) # target
            print("action_e:{}".format(action_e))
            
            print("after get_action")

            s_, reward, done, unit_size_, e_unit_size_ = env.step([action, action_e])#执行完动作后的时间，time2
            print("after step actions")
            screen_my_n = s_['screen_my']
            screen_enemy_n = s_['screen_enemy']

            if reward is not None:
                rewards.append(reward)
                for i in range(len(env.units_e_id)):
                    # print("i:{},replay_buffer.size:{}".format(i,replay_buffer.size))
                    replay_buffer.add_transition(screen_enemy[i], action_e[i], screen_enemy_n[i], 0, 1,14,14,[])

                
            var *= 0.9999
            screen_my = screen_my_n
            screen_enemy = screen_enemy_n

            current_step += 1
            if reward is not None:
                cumulative_reward += reward
        if epi_flag:
            episodes += 1
            if bool(env.state['battle_won']):
                battles_won += 1
            #--------------------------begin train---------------------------------
            print("begin train!")
            for _ in range(0,train_round):
                iters_num = int(replay_buffer.size * 2 / 3 / batch_size)
                print("iters_num:{}".format(iters_num))
                replay_buffer.shuffle()
                for _ in range(0,iters_num):
                    state_action_r,action_r,state_action_next_r,isterminal_r,rewards_r, command_size_r, command_size_next_r,dead_e_index_r = replay_buffer.get_sample(min(batch_size,replay_buffer.size))
                    loss,acc_rate = dqn.learn_with_one_episode(s=state_action_r,a=action_r,s_=state_action_next_r,isterminal=isterminal_r,r=rewards_r,command_size=command_size_r)  
                print("loss:{},acc_rate:{}".format(loss,acc_rate))
            print("end train!")
            #--------------------------end train---------------------------------
            wf(str(battles_won) + '\n' + str(episodes), 0)
            wf(str(np.mean(rewards)) + '\n' + str(episodes), 1)
            print('episodes:', episodes, ', win:', battles_won, ', mean_reward:', np.mean(rewards))
            if episodes % CONFIG.episode_to_reset_win == 0:
                win_rate[episodes] = battles_won / CONFIG.episode_to_reset_win
                print('win rate:', win_rate[episodes])
                battles_won = 0


        #print cumulative_reward
        if episodes % CONFIG.episode_to_save == 0 and episodes != 0:
            dqn.saver.save(sess, args.result + './model/model.ckpt', global_step=episodes)
            print("save model to {}".format(args.result + './model/model.ckpt'))



