#coding=utf8
from __future__ import division
import math
import pickle
import matplotlib.pyplot as plt
from gym_FPS.envs.starcraft.Config import *
import numpy as np

CONFIG = Config()

def get_degree(x1, y1, x2, y2):
    radians = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(radians)


def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_position(degree, distance, x1, y1):
    theta = math.pi / 2 - degree
    return x1 + distance * math.sin(theta), y1 + distance * math.cos(theta)


def print_progress(episodes, wins):
    print("Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
        episodes, wins, wins / (episodes + 1E-6)))


"""Return: closest enemy id"""
def get_closest(x1, y1, enemies):
    min_dist = 9999999
    id = -1
    for uid, ut in enemies.items():
        dist = get_distance(ut['POSITION'][0], ut['POSITION'][1], x1, y1)
        if dist < min_dist:
            id = uid
            min_dist = dist
    return id,min_dist

def maxmin_distance(units_my, units_enemy):
    max_dist = 0
    nb_us = 0
    nb_them = 0
    for _, f in units_my.items():
        min_dist = 1000
        nb_us = nb_us + 1
        for _, ef in units_enemy.items():
            dist = np.linalg.norm(np.array([f['POSITION'][0], f['POSITION'][2]]) - np.array([ef['POSITION'][0], ef['POSITION'][2]]))
            if dist < min_dist:
                min_dist = dist
        if min_dist > max_dist:
            max_dist = min_dist
        #print('max_dist',max_dist)
    for _, _ in units_enemy.items():
        nb_them = nb_them + 1
    if max_dist > CONFIG.max_dist and nb_them > 0 and nb_us > 0:
        return True
    else:
        return False


#TODO 计算最近的四个敌人
def get_enemy(x, y, enemies):
    pass

def get_rawout(a, b, c, d):
    x = 0
    y = 0
    try:
        y = math.sqrt(((a - c)**2 + (b - d)**2)/256) - 1
        tmp = math.asin((a - c)/(16*(y + 1)))
        x = 0.5 - (tmp / math.pi)
    except Exception as e:
        print(e)
    return x, y

# TODO
# @return:enemies list sorted by health
def get_weakest(enemies):
    pass

def save(episodes, agent, avg_rewards, episode_record,
         avg_loss, loss_sum, win_rate, battles_won, win_episode):
    if episodes % CONFIG.episode_to_save_reward == 0 and episodes != 0:
        dumpReward(avg_rewards, episode_record, 'reward' + str(episodes) + '.pkl')
        dumpLoss(avg_loss, loss_sum, episode_record, 'loss' + str(episodes) + '.pkl')
        win_rate.append(battles_won / CONFIG.episode_to_save_reward)
        win_episode.append(episodes)
        battles_won = 0

    if episodes % CONFIG.episode_to_save_win == 0 and episodes != 0:
        dumpWin(win_rate, win_episode, 'win' + str(episodes) + '.pkl')

    if episodes % CONFIG.episode_to_save_model == 0 and episodes != 0:
        agent.saver.save(agent.sess, CONFIG.model_dir + 'model.ckpt', global_step=episodes)
    return battles_won

def dumpReward(rewards, episodes, name):
    dump = []
    dump.append(rewards)
    dump.append(episodes)
    output = open(CONFIG.reward_dir + name, 'wb')
    pickle.dump(dump, output)
    output.close()

def dumpWin(win, episodes, name):
    dump = []
    dump.append(win)
    dump.append(episodes)
    output = open(CONFIG.win_dir + name, 'wb')
    pickle.dump(dump, output)
    output.close()

def dumpLoss(avg_loss, loss_sum, episodes, name):
    dump = []
    dump.append(avg_loss)
    dump.append(loss_sum)
    dump.append(episodes)
    output = open(CONFIG.loss_dir + name, 'wb')
    pickle.dump(dump, output)
    output.close()

def loadReward(name):
    pkl_file = open(CONFIG.reward_dir + name, 'rb')
    dump = pickle.load(pkl_file)
    return dump[0], dump[1]

def loadWin(name):
    pkl_file = open(CONFIG.win_dir + name, 'rb')
    dump = pickle.load(pkl_file)
    return dump[0], dump[1]

def loadLoss(name):
    pkl_file = open(CONFIG.loss_dir + name, 'rb')
    dump = pickle.load(pkl_file)
    return dump[0], dump[1], dump[2]

def plotReward(x, y):
    plt.xlim((0, 50))
    plt.plot(x, y, color='red', linestyle='dashed', marker='o')
    plt.show()

def get_target(dict, id):
    try:
        return dict[id].x, dict[id].y
    except Exception as e:
        print(e)


def list2str(l, split_char=','):
    if type(l) == int:
        l = [l]
    s = split_char.join([str(i) for i in l])
    return s

def dict2str(d, split_char='|', equal_char=':'):
    l = []
    for (key, values) in d.items():
        l.append('%s:%s' % (key, values))

    s = split_char.join(l)
    return s

def str2list(s):
    s = s.split('=')[-1]
    l = [float(i) for i in s.split(',')]
    return l

def str2dict(s, split_char='|', equal_char=':', left_type=None):
    '''
    left_type:id_list时需将key转为int类型
    '''
    if s[-1] == '|':
        s = s[:-1]
    d = dict()
    ss = s.split('=')
    for pair in ss[-1].split(split_char):
        (key, value) = pair.split(equal_char)
        if 'int' == left_type:
            key = int(key)
        if value.find(',') > -1:
            value = str2list(value)
        else:
            try:
                value = float(value)
            except:
                pass
        d[key] = value
    return ss[0], d

def get_simple_id_list(l):
    '''
    将一个常规id_list转化为简写的str
    '''
    l = list(l)
    l.sort()
    s = ''
    st, ed = 0, 0
    while ed < len(l):
        while ed + 1 < len(l) and ed - st + 1 == l[ed + 1] - l[st]:
            ed += 1
        if st == ed:
            s += '%d,' % l[st]
        else:
            s += '%d-%d,' % (l[st], l[ed])
        ed += 1
        st = ed
    return s[:-1]

def simple_list_to_full(s):
    '''
    将间歇的id_list转为常规list
    输入:s  str
    return list
    '''
    l = s.split(',')
    result = []
    for uid in l:
        if uid.find('-') > -1:
            (st, ed) = uid.split('-')
            for i in range(int(st), int(ed) + 1):
                result.append(i)
        else:
            result.append(int(uid))

    return result

def get_ai(d):
    ##############################
    #s='<%s/>'%''
    return ''

def get_action(d):
    '''
    动作字典转为指令
    '''
    for (key, value) in d.items():
        if 'objid_list' == key:
            value = get_simple_id_list(value)
        elif 'ai' == key:
            value = get_ai(value)
        else:
            value = '[%s]' % list2str(value, split_char='|')


def pos2mapid(pos):
    '''
    真实坐标转5*5地图坐标
    '''
    x = (pos[0] + 275) // 50
    x = min(4, max(0, x))
    y = (pos[2] + 120) // 60
    y = min(4, max(0, y))
    return int(x), int(y)

def get_dis(pos1, pos2, including_h=False):
    '''
    计算两点间距离
    '''
    dis = (pos1[0] - pos2[0]) ** 2 + (pos1[2] - pos2[2]) ** 2
    if including_h:
        dis += (pos1[1] - pos2[1]) ** 2
    return dis ** 0.5

def normalize(x, y):

    s = (x ** 2 + y ** 2) ** 0.5
    if s < 0.01:
        return x, y
    x /= s
    y /= s
    return x, y

def get_units_center(units):
    x = 0
    y = 0
    nunits = 0
    for uid, feats in units.items():
        x += feats['POSITION'][0]
        y += feats['POSITION'][2]
        nunits += 1
    if nunits == 0:
        print("error")
    return float(x) / nunits, float(y) / nunits

