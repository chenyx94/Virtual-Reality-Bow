# coding: utf-8

import math, time, random
import numpy as np
from gym import spaces
from .import FPS_env as fps

from ..utils import *

class DemoEnv3(fps.FPSEnv):
    def __init__(self,):
        super(DemoEnv3, self).__init__()

    def _step(self, action):
        print('receive action:%d'%action)
        self.state = self.get_state1()

        if action == 1:#围一圈保护移动
            self.move_alert()
        elif action == 2:#人墙
            self.move_to_ahead()
        elif action == 3:#自由攻击
            #self.origin_ai(team_id=1)
            #self.search_enemy_attack()
            self.super_attack()
        elif action == 4:#包围
            self.attack_surround()
        elif action == 5:#跟随不攻击
            self.move_follow()
        return self.state, 0, self._check_done(), ''

    def _reset(self, ):
        self.new_episode()
        #self.playerai()
        #self.get_game_variable()
        return self.get_state1()

    def _check_done(self):
        hp1, hp2 = 0, 0
        for uid, unit in self.states.items():
            if unit['TEAM_ID'] > 0:
                hp1 += max(0, unit['HEALTH'])
            else:
                hp2 += max(0, unit['HEALTH'])

        return hp1 * hp2 == 0



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
            state['cur_x_%s'%name] = -999 if len(units) == 0 else np.mean([u['POSITION'][0] for uid, u in units.items()])
            state['cur_y_%s'%name] = -999 if len(units) == 0 else np.mean([u['POSITION'][2] for uid, u in units.items()])
            if state['cur_x_%s'%name] == -999:
                state['cur_map_x_%s' % name], state['cur_map_y_%s' % name] = -1, -1
            else:
                state['cur_map_x_%s' % name], state['cur_map_y_%s' % name] = pos2mapid([state['cur_x_%s' % name], -1, state['cur_y_%s' % name]])

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


        self.enemy_nearby = self.get_enemy_nearby() #enemy 暂定为附近的敌人
        units_enemy = dict()
        for uid, dis in self.enemy_nearby.items():
            units_enemy[uid] = self.states[uid]
        get_state(state, units_total_enemy, 'enemy_total')
        get_state(state, units, 'I')
        get_state(state, units_friend, 'friend')
        get_state(state, units_enemy, 'enemy')

        return state
