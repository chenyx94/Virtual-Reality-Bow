# coding: utf-8

import math, time, random
import numpy as np
from gym import spaces
from . import FPS_env as fps

from ..utils import *


class MapEnv(fps.FPSEnv):
    def __init__(self, ):
        super(MapEnv, self).__init__()
        self.target_mapid = [0, 0]
        self.is_battle = False
        self.enemy_nearby = dict()
        self.t = time.time()

    def _step(self, action):
        # super(DemoEnv, self).get_game_variable()
        hp1, hp2, hp3, hp4 = 0, 0, 0, 0
        for uid, unit in self.states.items():
            if unit['TEAM_ID'] > 0:
                hp1 += max(0, unit['HEALTH'])
            else:
                hp2 += max(0, unit['HEALTH'])

        self.target_mapid = [action // 5, action % 5]
        # self.map_move(team_id=1, target_map_pos=self.target_mapid)
        self.map_move(team_id=1, objid_list=[0], target_map_pos=self.target_mapid)
        #print("before check new frame")
        is_new_frame, s = self.check_frame()
        while not is_new_frame:
            time.sleep(1 / self.speedup)
            is_new_frame, s = self.check_frame()
            if len(self.enemy_nearby) > 0:
                enemy_nearby_id = list(self.enemy_nearby.keys())
                # print(enemy_nearby_id)
                for uid in self.team_member[1]:
                    target = random.choice(enemy_nearby_id)
                    # print(uid, target)
                    if uid not in self.attack_target.keys():
                        self.set_target_objid([uid], target)
                    elif self.states[target]['HEALTH'] <= 0:
                        self.set_target_objid([uid], target)
                    self.attack([uid])
            else:
                self.move_alert()
            #print("out check new frame loop")

        #print('new frame:%s' % s)
        for uid, unit in self.states.items():
            if unit['TEAM_ID'] > 0:
                hp3 += max(0, unit['HEALTH'])
            else:
                hp4 += max(0, unit['HEALTH'])

        reward = hp1 - hp3 - hp2 + hp4#hp1,2执行前 hp3,4执行后
        eps = 0.1
        done = hp3 * hp4 < eps
        #print(hp3,hp4,done)
        return self.transfer_state(self.get_state1()), reward, done, '' if not done else 'win' if hp4 == 0 else 'lose'

    def _reset(self, ):
        self.new_episode()
        self.playerai()
        # self.get_game_variable()
        play_pos = [-150,-1,47]
        enemy_pos = [-100, -1, -100]
        for i in range(5):
            play_pos[0] -= 1
            play_pos[2] -= 1
            enemy_pos[0] -= 1
            enemy_pos[2] -= 1
            self.add_obj(name="friend"+str(i),is_enemy=False,pos=play_pos,leader_objid=0,team_id=1)
            #self.add_obj(name="enemy"+str(i), is_enemy=True, pos=enemy_pos, leader_objid=-1, team_id=-1)

        self.add_observer([-150,-1,47], 200)
        return self.get_state1()

    def get_state1(self, team_id=1):
        '''
        获得原始状态  真实坐标的None暂时置-999  地图坐标None置-1
        我方 友军 所有敌人 附近的敌人
        *
        剩余人数 平均血量 平均实际坐标 平均地图坐标(5*5)  目标地图坐标
        没有做其他处理
        '''

        def get_state(state, units, name):
            state['live_number_%s' % name] = len(units)
            state['cur_x_%s' % name] = -999 if len(units) == 0 else np.mean(
                [u['POSITION'][0] for uid, u in units.items()])
            state['cur_y_%s' % name] = -999 if len(units) == 0 else np.mean(
                [u['POSITION'][2] for uid, u in units.items()])
            if state['cur_x_%s' % name] == -999:
                state['cur_map_x_%s' % name], state['cur_map_y_%s' % name] = -1, -1
            else:
                state['cur_map_x_%s' % name], state['cur_map_y_%s' % name] = pos2mapid(
                    [state['cur_x_%s' % name], -1, state['cur_y_%s' % name]])
            if name == 'I':
                state['target_map_x_%s' % name] = self.target_mapid[0]
                state['target_map_y_%s' % name] = self.target_mapid[1]

        state = dict()
        units = dict()
        units_friend = dict()
        units_total_enemy = dict()
        for unitid, unit in self.states.items():
            if unit['TEAM_ID'] == team_id:
                units[unitid] = unit
            elif unit['TEAM_ID'] < 0:
                units_total_enemy[unitid] = unit
            else:
                units_friend[unitid] = unit

        self.enemy_nearby = self.get_enemy_nearby()  # enemy 暂定为附近的敌人
        units_enemy = dict()
        for uid, dis in self.enemy_nearby.items():
            units_enemy[uid] = self.states[uid]
        get_state(state, units_total_enemy, 'enemy_total')
        get_state(state, units, 'I')
        get_state(state, units_friend, 'friend')
        get_state(state, units_enemy, 'enemy')

        return state

    def check_frame(self, ):
        '''
        检查是否为新的一帧
        '''
        try:
            unit0 = self.states[0]
        except:
            print(__name__,self.states)
        mapid = list(pos2mapid(unit0['POSITION']))
        t = time.time()
        if t - self.t > 30:
            self.t = t
            return True, 'time_out'
        if mapid != self.mapid:
            self.mapid = mapid
            self.frame += 1
            self.t = t
            return True, 'new_map_pos'
        self.enemy_nearby = self.get_enemy_nearby()
        if len(self.enemy_nearby) > 0 and self.is_battle == False:
            self.frame += 1
            self.is_battle = True
            self.t = t
            return True, 'find_enemy'
        if self.is_battle and len(self.enemy_nearby) == 0:
            self.frame += 1
            self.is_battle = False
            self.t = t
            return True, 'battle_end'
        if mapid == self.target_mapid:
            self.frame += 1
            self.t = t
            return True, 'arrive'
        return False, ''

    def transfer_state(self,state):
        """
        25*5位置特征
        4 归一化存活人数
        129维
        """

        def get_map_id(x, y):
            return x + y * 5

        onehot_cur_map_I = np.zeros(25, dtype='int')
        onehot_cur_map_enemy = np.zeros(25, dtype='int')
        onehot_cur_map_enemy_total = np.zeros(25, dtype='int')
        onehot_cur_map_friend = np.zeros(25, dtype='int')
        onehot_target_map_I = np.zeros(25, dtype='int')

        total_number = len(self.units)
        normalized_live_number = np.zeros(4)

        if total_number > 0:
            normalized_live_number[0] = state['live_number_I']
            normalized_live_number[1] = state['live_number_enemy']
            normalized_live_number[2] = state['live_number_enemy_total']
            normalized_live_number[3] = state['live_number_friend']
            normalized_live_number = normalized_live_number * 1.0 / total_number

        onehot_cur_map_I[get_map_id(state['cur_map_x_I'], state['cur_map_y_I'])]
        onehot_cur_map_enemy[get_map_id(state['cur_map_x_enemy'], state['cur_map_y_enemy'])]
        onehot_cur_map_enemy_total[get_map_id(state['cur_map_x_enemy_total'], state['cur_map_y_enemy_total'])]
        onehot_cur_map_friend[get_map_id(state['cur_map_x_friend'], state['cur_map_y_friend'])]
        onehot_target_map_I[get_map_id(state['target_map_x_I'], state['target_map_y_I'])]

        onehot_map_feature = np.hstack((onehot_cur_map_I, onehot_cur_map_enemy, onehot_cur_map_enemy_total,
                                        onehot_cur_map_friend, onehot_target_map_I))
        concat_feature = np.hstack((onehot_map_feature, normalized_live_number))
        return concat_feature


