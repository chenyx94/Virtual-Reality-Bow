# coding: utf-8

import numpy as np
import math, random, threading
import gym
import time, os

from ..utils import *
from ..client import Client
from .starcraft import Config


class FPSEnv(gym.Env):
    def __init__(self, ):
        # self.set_env()
        # self.playerai()
        self.thread_flag = True
        try:
            CONFIG = Config.Config()
            os.popen(CONFIG.game_dir)
            time.sleep(CONFIG.wait_for_game_start)
        except:
            print('自动打开失败')

    def _step(self, action):
        raise NotImplementedError

    def _reset(self, ):
        raise NotImplementedError

    def _action_space(self):
        """Returns a space object"""
        raise NotImplementedError

    def _observation_space(self):
        """Returns a space object"""
        raise NotImplementedError

    def _make_commands(self, action):
        """Returns a game command list based on the action"""
        raise NotImplementedError

    def _make_observation(self):
        """Returns a observation object based on the game state"""
        raise NotImplementedError

    def _compute_reward(self):
        """Returns a computed scalar value based on the game state"""
        raise NotImplementedError

    def _check_done(self):
        """Returns true if the episode was ended"""
        # raise NotImplementedError
        return False

    def _get_info(self):
        """Returns a dictionary contains debug info"""
        return {}

    def render(self, mode='human', close=False):
        pass

    def set_env(self, SEVERIP='127.0.0.1', SERVERPORT=5123, client_DEBUG=False, env_DEBUG=False, speedup=1):
        '''
        配置env
        '''
        self.IP = SEVERIP
        self.port = SERVERPORT
        self.client_DEBUG = client_DEBUG
        self.client = Client(SEVERIP=SEVERIP, SERVERPORT=SERVERPORT, DEBUG=client_DEBUG)
        # self.client2 = Client(SEVERIP=SEVERIP, SERVERPORT=SERVERPORT, DEBUG=client_DEBUG, name='client2')
        self.units = dict()
        self.states = dict()
        self.episode_id = 0
        self.mapid = [0, 1]
        self.frame = 0
        # self.stepaaa = 0
        self.DEBUG = env_DEBUG

        self.speedup = speedup

        self.team_target = dict()
        self.team_member = dict()

        self.attack_target = dict()
        self.refs = dict()
        self.mark_id = 0
        self.path_id = 0
        self.episode_id = 1

    def get_objid_list(self, name=1, pos=0):
        '''
        获取所有人id  名字和坐标可选
        return dict
        '''
        cmd = 'cmd=get_objid_list`name=%d`pos=%d' % (name, pos)
        self.client.send(cmd)
        while len(self.client.objid_list) == 0:
            time.sleep(0.1)
            self.client.send(cmd)
        s = self.client.objid_list.split('`')
        try:
            _, self.units = str2dict(s[1], left_type='int')
        except:
            print(s)
            raise
        if self.DEBUG:
            print(self.units)
        return self.units

    def get_pos(self, ):
        '''
        根据self.states获取所有人坐标
        '''
        pos = dict()
        for uid, unit in self.states.items():
            pos[uid] = unit['POSITION']

        return pos

    def get_game_variable(self, ):
        '''
        获取units所有状态 但不包括弹药数 敌我标记（暂时用team_id代替 -1为敌人 正数为我方）
        objid_list 为 'all' 或 list
        值会保存在self.states中
        return dict{id:dict{key:value}}
        '''
        while True:
            team_member = dict()
            objid_list = self.get_objid_list()
            objid_list = get_simple_id_list(objid_list.keys())

            states = self.states.copy()
            for key in states.keys():
                states[key]['LAST_POSITION'] = states[key]['POSITION']

            self.client.send('cmd=get_game_variable`objid_list=%s' % (objid_list))
            s = self.client.game_variable
            count111 = 0
            while len(s) == 0:
                time.sleep(0.05)
                s = self.client.game_variable
                count111 += 1
                if count111 % 10 == 0:
                    self.client.send('cmd=get_game_variable`objid_list=%s' % (objid_list))
            for key in states.keys():
                states[key]['HEALTH'] = 0
            for ss in s.split('`')[1:]:
                unit_id, d = str2dict(ss)
                if int(unit_id) not in states.keys():
                    states[int(unit_id)] = dict()
                for key, value in d.items():
                    states[int(unit_id)][key] = value
                states[int(unit_id)]['TIME'] = time.time()
                # 每个队有哪些人
                try:
                    tid = int(d['TEAM_ID'])
                except:
                    print(d)
                    raise
                if tid not in team_member.keys():
                    team_member[tid] = []
                team_member[tid].append(int(unit_id))

            self.team_member = team_member
            self.states = states.copy()
            if self.DEBUG:
                print(self.states)
            time.sleep(0.5)
        return  # self.states

    def new_episode(self, save=1, replay_file=None, speedup=1, disablelog=0, scene_name='Simple'):
        '''
        新的一局 replay_file为回放的文件名
        游戏的speedup请在IsLand.xml中设置
        '''
        # print('new_episode')
        self.episode_id += 1
        self.client.notify = []
        self.team_target = dict()
        # cmd = 'cmd=new_episode'
        cmd = 'cmd=new_episode`episode_id=%d`save=%d`disablelog=%d`scene_name=%s' % (self.episode_id, save, disablelog, scene_name)
        if str == type(replay_file):
            cmd += '`replay_file=%s' % replay_file
        self.client.send(cmd)
        # s = self.client.receive()
        # self.playerai()
        time.sleep(3.0)
        self.states = dict()
        self.client.game_variable = ''
        self.client.check_pos = ''
        self.client.objid_list = ''
        if self.thread_flag:
            self.thread_flag = False
            self.t1 = threading.Thread(target=self.get_game_variable)
            self.t1.start()

        # self.get_game_variable()
        # if s.find('success')==-1:
        #   self.reset()
        #
        return  # s

    def restart(self, sleep_time=10, port=-1):
        '''
        在指定端口上完全重启软件
        '''
        if port < 0:
            self.port = self.port // 2 * 4 + 1 - self.port
        else:
            self.port = port
        cmd = 'cmd=restart`port=%d' % self.port
        self.client.send(cmd)
        time.sleep(sleep_time)
        self.set_env(self.IP, SERVERPORT=self.port, client_DEBUG=self.client_DEBUG)

    def playerai(self, ):
        '''
        主角进入ai模式
        '''
        cmd = 'cmd=common`type=playerai'
        self.client.send(cmd)
        # s = self.client.receive()
        return  # s

    def ailog(self, objid):
        '''
        输出某人的日志 reset不会改变该状态
        '''
        cmd = 'cmd=common`type=ailogobj`objid=%d' % objid
        self.client.send(cmd)
        #s = self.client.receive()
        return # s

    def make_action(self, d):
        '''
        TODO dict
        做一个自定义动作
        '''
        if dict == type(d):
            s = get_action(d)
        else:
            s = d
        cmd = 'cmd=make_action`' + s
        self.client.send(cmd)
        # s = self.client.receive()
        return  # s

    def add_obj(self, name, is_enemy, pos, leader_objid, team_id, dir=[0, 0, 0], model_name='DefaultAI', weapon='m4'):
        '''
        添加一个人
        '''
        is_enemy = 'enemy' if is_enemy else 'comrade'
        cmd = 'cmd=add_obj`type=%s`pos=%s`dir=%s`ai=%s`leader_objid=%d`name=%s`team_id=%d`weapon=%s' \
              % (is_enemy, list2str(pos), list2str(dir), model_name, int(leader_objid), name, int(team_id), weapon)
        self.client.send(cmd)
        # s = self.client.receive()
        # if s.find('success')==-1:
        #   self.reset()
        return  # s

    def add_obj_list(self, name, pos, leader_objid, team_id, width, num, is_enemy=False, dir=[0, 0, 0],
                     model_name='DefaultAI'):
        '''
        在一个区域内随机添加一堆人 会重名
        '''
        is_enemy = 'enemy' if is_enemy else 'comrade'
        cmd = 'cmd=add_obj_list`type=%s`pos=%s`width=%d`num=%d`dir=%s`ai=%s`leaderObjID=%d`name=%s`team_id=%d' \
              % (is_enemy, list2str(pos), width, num, list2str(dir), model_name, int(leader_objid), name, int(team_id))
        self.client.send(cmd)
        # s = self.client.receive()
        # if s.find('success')==-1:
        #   self.reset()

        return  # s

    def check_pos(self, pos, objid=-1):
        '''
        检查目标点对于某人来说是否可达 -1为随机一人
        '''
        cmd = 'cmd=check_pos`pos=%s`objid=%d' % (list2str(pos), int(objid))
        self.client.send(cmd)
        # s = self.client.receive()
        '''
        if s.find('fail') > -1:            
            if self.DEBUG:
                print('check_pos fail')
            raise
        return s.find('1') > -1
        '''

    def move(self, destPos, objid_list='all', group='group1', auth='normal', pos='replace', walkType='walk',
             reachDist=6, maxDoTimes='', team_id=None, ):
        '''
        强行移动不受其他因素干扰 挨打不还手用于撤退 队形变换
        '''
        if type(objid_list) == int:
            objid_list = list2str([objid_list])
        if objid_list == 'all':
            objid_list = get_simple_id_list(self.units.keys())
        else:
            objid_list = get_simple_id_list(objid_list)
        cmd = 'objid_list=%s`' % objid_list
        if team_id != None:
            cmd += 'team_id=%d`' % team_id
        if auth != 'top':
            if maxDoTimes == '':
                maxDoTimes = 1
            maxDoTimes = " maxDoTimes='%d'" % maxDoTimes
        else:
            if type(maxDoTimes) == int:
                maxDoTimes = " maxDoTimes='%d'" % maxDoTimes
        cmd += "auth=%s`group=%s`pos=%s`ai=<action name='MoveToPosAct' destPos='%s' walkType='%s' reachDist='%d'%s/>" \
               % (auth, group, pos, list2str(destPos), walkType, reachDist, maxDoTimes)
        s = self.make_action(cmd)
        return s

    def add_patrol_path(self, pos_list, objid, noteam=0, noleader=1):
        '''
        添加巡逻路径
        '''
        if type(pos_list[0]) != list:
            if self.DEBUG:
                print('type of pos_list should be list[][]')
            return 'error'
        pos_list = '|'.join([list2str(pos) for pos in pos_list])
        objid = list2str(objid)
        cmd = 'cmd=add_patrol_path`pos_list=%s`objid=%s`noleader=%d`noteam=%d' % (pos_list, objid, noleader, noteam)
        self.client.send(cmd)
        # s = self.client.receive()
        '''
        if s.find('fail') > -1:
            if self.DEBUG:
                print(s)
            raise
        return s
        '''

    def add_map_mark(self, pos, marktype='mark', blinking_time=-1, lead_obj_id=0):
        '''
        小地图添加标记
        type=mark|arrow|blinking|focus
        '''
        cmd = 'cmd=add_map_mark`pos=%s`type=%s' % (list2str(pos), marktype)
        if marktype == 'blinking' or marktype == 'focus':
            if blinking_time > 0:
                cmd += '`blinking_time=%d' % blinking_time
        elif marktype == 'arrow':
            cmd += '`lead_obj_id=%d' % lead_obj_id
        self.client.send(cmd)
        self.mark_id += 1
        # s = self.client.receive()
        '''
        if s.find('success') < 0:
            print('add mark failed')
            #raise
        return s
        '''
        return self.mark_id

    def remove_map_mark(self, mark_id):
        '''
        根据id清除mark
        '''
        cmd = 'cmd=remove_map_mark`id=%d' % mark_id
        self.client.send(cmd)

    def draw_pathline(self, pos_list):
        '''
        绘制一条提示路径
        '''
        pos_list = '|'.join([list2str(l) for l in pos_list])
        cmd = 'cmd=draw_pathline`pos_list=%s' % pos_list
        self.client.send(cmd)
        self.path_id += 1
        return self.path_id

    def remove_pathline(self, path_id):
        '''
        根据id清除path
        '''
        cmd = 'cmd=remove_pathline`id=%d' % path_id
        self.client.send(cmd)

    def add_observer(self, pos, radius):
        '''
        用于清除迷雾
        '''
        cmd = 'cmd=add_observer`pos=%s`radius=%d' % (list2str(pos), radius)
        self.client.send(cmd)

    def add_chat(self, msg, obj_id, close_time=5):
        self.show_ui_win('ChatWin', 1)
        cmd = 'cmd=add_chat`msg=%s`obj_id=%d' % (msg, obj_id)
        if close_time > 0:
            cmd += '`close_time=%d' % close_time
        self.client.send(cmd)

    def select_obj(self, objid, is_select=1):
        '''
        脚底高亮 重新选择、取消选择
        '''
        cmd = 'cmd=select_obj`obj_id=%d`is_select=%d' % (objid, is_select)
        self.client.send(cmd)

    def set_task(self, msg):
        '''
        设置任务面板
        '''
        self.show_ui_win('TaskWin', 1)
        cmd = 'cmd=set_task`msg=%s' % (msg)
        self.client.send(cmd)
        threading.Thread(target=self.wait_close, args=('TaskWin', 5,)).start()

    def wait_close(self, name, wait_time=5, ):
        time.sleep(wait_time)
        self.show_ui_win(name, 0)


    def show_ui_win(self, name, is_show=1):
        '''
        显示ui界面
        name=[HelpWin|CtrlWin|ReplayWin|InfoWin|StatusWin|ChatWin|TaskWin]
        '''
        cmd = 'cmd=show_ui_win`name=%s`is_show=%d' % (name, is_show)
        self.client.send(cmd)

    def watch_obj(self, obj_id, is_watch=1):
        '''
        附身观察
        '''
        cmd = 'cmd=watch_obj`obj_id=%d`is_watch=%d' % (obj_id, is_watch)
        self.client.send(cmd)

    def is_arrived(self, objid, target_pos, dis=-1):
        '''
        是否到达目标点  dis为阈值
        return dis=-1时返回距离 否则返回距离是否小于dis
        '''
        p = self.states[objid]['POSITION']
        d = math.pow(p[0] - target_pos[0], 2) + math.pow(p[2] - target_pos[2], 2)
        if self.DEBUG:
            print(d, d <= dis * dis)
        if dis < 0:
            return math.sqrt(d)
        return d <= dis * dis

    def create_team(self, leader_objid, member_objid_list, team_id):
        '''
        编队  队员必须提前存在
        '''
        cmd = 'cmd=create_team`leader_objid=%d`member_objid_list=%s`team_id=%d' % (
            leader_objid, list2str(member_objid_list), team_id)
        self.client.send(cmd)
        # s = self.client.receive()
        return  # s

    def search_enemy_attack(self, objid_list='all', team_id=1, auth='normal', group='group1', pos='replace'):
        '''
        搜索敌人并攻击
        搁置!!!!!!!!!!!!!!
        '''
        if type(objid_list) == int:
            objid_list = list2str([objid_list])
        elif objid_list == 'all':
            objid_list = get_simple_id_list(self.units.keys())
        else:
            objid_list = list2str(objid_list)
        cmd = "cmd=make_action`objid_list=%s`team_id=%d`auth=%s`group=%s`pos=%s`ai=<check name='CheckTimeChk' interval='0'><action name='ShootAct'/><action name='SearchEnemyAct'/><check/>" % \
              (objid_list, team_id, auth, group, pos)
        # print(cmd)
        self.client.send(cmd)
        # s = self.client.receive()
        return  # s

    def super_attack(self, team_id=1, mindis=50):
        '''
        超搜索距离攻击！！！
        '''
        enemy_nearby = self.get_enemy_nearby(mindis=mindis)
        if len(enemy_nearby) == 0:
            return 
        for uid in self.team_member[team_id]:
            eid = random.sample(enemy_nearby.keys(), 1)[0]
            self.can_attack_move([uid], destPos=self.states[eid]['POSITION'], team_id=1, walkType='run')

    def can_attack_move(self, objid_list, destObjID='', destObj='', destPos='', team_id=None, auth='normal',
                        group='group1', pos='replace', walkType='walk', reachDist=6):
        '''
        移动时检查能否攻击 实际测试时经常反应慢一拍
        移动优先级
        1  destObjID 目标ID
        2  destObj   target or leader
        3  destPos   目标坐标
        '''
        if destPos == '' and destObjId == '' and destObj == '':
            print('need destPos, destObjid or destObj at least one')
            raise ValueError
        if type(objid_list) == int:
            objid_list = list2str([objid_list])
        elif objid_list == 'all':
            objid_list = get_simple_id_list(self.units.keys())
        else:
            objid_list = list2str(objid_list)

        cmd = 'cmd=make_action`objid_list=%s' % objid_list
        if type(team_id) == int:
            cmd += '`team_id=%d' % team_id
        cmd += '`auth=%s`group=%s`pos=%s`ai=' % (auth, group, pos)
        cmd += '<check name="CheckTimeChk" interval="0">'
        cmd += '<check name="SearchEnemyAct"><action name="ShootAct"/>'  # shootcd 将在下个版本中生效
        cmd += '<action name="MoveToPosAct" destObj="target" walkType="run" reachDist="6"/></check>'
        # cmd += "<check name='CanAttackTargetAct'><action name='ShootAct'/></check>"
        cmd += '<action name="MoveToPosAct" destObjID="%s" destObj="%s" destPos="%s" walkType="%s"  reachDist="%d"/></check>' % (
            destObjID, destObj, list2str(destPos), walkType, reachDist)
        if self.DEBUG:
            print(destPos)
            print(cmd)
        # cmd += "`maxDoTimes='%d'" % maxDoTimes
        self.client.send(cmd)
        # s = self.client.receive()
        return  # s

    def attack(self, objid_list, auth='normal', pos='replace'):
        '''
        攻击  若没有settargetact或searchenemyact指定target则无效
        '''
        cmd = "cmd=make_action`objid_list=%s`auth=%s`group=group1`pos=%s`ai=" % (list2str(objid_list), auth, pos)
        cmd += '<check name="CheckTimeChk" interval="0"><action name="ShootAct"/><action name="MoveToPosAct" destObj="target" walkType="run"  reachDist="12"/></check>'
        # print(cmd)
        self.client.send(cmd)
        # s = self.client.receive()
        return  # s

    def set_target_objid(self, objid_list, targetObjID, auth='replace'):
        '''
        强行指定攻击目标 还需再调attack才会攻击
        '''
        #cmd = "cmd=make_action`objid_list=%s`auth=%s`group=group1`pos=head`ai=" % (list2str(objid_list), auth)
        #cmd += "<action name='SetTargetAct' targetObjID='%d'/>" % targetObjID
        cmd = 'cmd=set_target`obj_id=%s`target_id=%d' % (list2str(objid_list), targetObjID)
        self.client.send(cmd)
        # s = self.client.receive()
        return  # s

    def get_enemy_nearby(self, team_id=1, mindis=30):
        '''
        return dict 附近敌人的objid:dis
        TODO mindis设置多少？
        return dict{id:dist}
        '''
        # print(self.team_member)
        enemy_nearby = dict()
        for enemyid, enemy_unit in self.states.items():
            if enemy_unit['TEAM_ID'] > 0 or enemy_unit['HEALTH'] <= 0:
                continue
            for unitid in self.team_member[team_id]:
                unit = self.states[unitid]
                dis = get_dis(enemy_unit['POSITION'], unit['POSITION'])
                if dis < mindis:
                    enemy_nearby[enemyid] = dis
                    break

        return enemy_nearby

    def map_move(self, target_map_pos, objid_list='all', team_id=None, can_attack=True):
        '''
        按照5*5坐标移动
        '''
        x = [-250, -200, -150, -100, -60]
        y = [-60, -10, 20, 100, 120]
        map_pos = [[[0, 0, 0]] * 5 for i in range(5)]
        for j in range(5):
            for i in range(5):
                map_pos[i][j] = [x[i], -1, y[j]]

        map_pos[0][2] = [-230, -1, 20]
        map_pos[1][4] = [-180, -1, 120]

        if can_attack:
            s = self.can_attack_move(objid_list=objid_list, team_id=team_id,
                                     destPos=map_pos[target_map_pos[0]][target_map_pos[1]], )
        else:
            s = self.move(objid_list=objid_list, team_id=team_id,
                          destPos=map_pos[target_map_pos[0]][target_map_pos[1]], )

            # return s

    def move_target(self, objid_list='all', target_id=0, team_id=1, walkType='run'):
        '''
        向某个人集中, 默认为第一队向主角跑步集中
        '''
        return self.can_attack_move(objid_list, destObj=target_id, team_id=team_id, walkType=walkType)

    def move_alert(self, team_id=1, capital_id=0, auth='normal', group='group1', walkType='run', dist=4, dist2=10,
                   reachDist=1):
        '''
        警戒移动
        '''
        # self.get_game_variable()
        capital_pos = self.states[capital_id]['POSITION']
        # angle0 = (90 - self.states[capital_id]['ANGLE']) / 180 * np.pi
        delta_x = self.states[capital_id]['POSITION'][0] - self.states[capital_id]['LAST_POSITION'][0]
        delta_x = max(-1, min(delta_x, 1))
        delta_y = self.states[capital_id]['POSITION'][2] - self.states[capital_id]['LAST_POSITION'][2]
        delta_y = max(-1, min(delta_y, 1))
        delta_x, delta_y = normalize(delta_x, delta_y)

        count = 0
        team_number = len(self.team_member[team_id]) - 1
        for uid in self.team_member[team_id]:
            if uid == capital_id:
                continue
            count += 1
            angle1 = count / team_number * 2 * np.pi
            destPos = [capital_pos[0] + dist * np.cos(angle1) + dist2 * delta_x, -1,
                       capital_pos[2] + dist * np.sin(angle1) + dist2 * delta_y]
            # print(uid, angle1/np.pi*180, capital_pos, destPos)
            self.move(destPos=destPos, objid_list=[uid], auth=auth, group=group, pos='replace', walkType=walkType,
                      reachDist=reachDist)

    def move_to_ahead(self, objid_list='all', team_id=1, capital_id=0, auth='normal', group='group1', walkType='run',
                      angle=None, dist=4, reachDist=2):
        '''
        挡住某人
        angle为阻挡方向
        TODO怎么个挡法？
        '''
        # self.get_game_variable()
        capital_pos = self.states[capital_id]['POSITION']
        if angle == None:
            #if capital_id in self.team_target.keys() and self.team_target[capital_id]['HEALTH'] > 0:
            #    target_id = self.team_target[capital_id]
            #    target_pos = self.states[target_id]['POSITION']
            #    angle = np.arctan2(target_pos[2] - capital_pos[2], target_pos[0] - capital_pos[0])
            enemy_nearby = self.get_enemy_nearby(mindis=40)
            if len(enemy_nearby) > 0:
                eid = list(enemy_nearby.keys())[0]
                target_pos = self.states[eid]['POSITION']
                print(target_pos, capital_pos)
                angle = np.arctan2(target_pos[2] - capital_pos[2], target_pos[0] - capital_pos[0])
            else:
                #print('need angle when capital have no attack target')
                #return
                angle = 0

        # print('angle%f'%angle)
        count = 0
        for uid in self.team_member[team_id]:
            if uid == capital_id:
                continue
            count += 1
            # self.attack(uid)
            if len(self.team_member[team_id]) % 2 == 1:
                angle1 = angle + (2 * ((count - 1) // 2) + 1) * (-1) ** count * 0.1
            else:
                angle1 = angle + count // 2 * (-1) ** count * 0.2
            destPos = [capital_pos[0] + dist * np.cos(angle1), -1, capital_pos[2] + dist * np.sin(angle1)]
            # print(destPos)
            self.move(destPos=destPos, objid_list=[uid], auth=auth, group=group, pos='replace', walkType=walkType,
                      reachDist=reachDist)
            threading.Thread(target=self.arrive_attack, args=(uid, destPos, 1, False)).start()

    def attack_surround(self, target_objid=-1, team_id=1, capital_id=0, dis=15):
        '''
        指定小队成员先包围再攻击
        暂定为半圆  整圆太容易死
        愚蠢的实现方法
        '''
        if target_objid < 0:
            enemy_nearby = self.get_enemy_nearby()
            if len(enemy_nearby) == 0:
                return
            target_objid = list(enemy_nearby.keys())[0]
        num = len(self.team_member[team_id]) - 1
        target_pos = self.states[target_objid]['POSITION']
        capital_pos = self.states[capital_id]['POSITION']
        angle = np.arctan2(capital_pos[2] - target_pos[2], capital_pos[0] - target_pos[0])
        count = 0
        # self.search_enemy_attack(auth='top')
        for uid in self.team_member[team_id]:
            if uid == capital_id:
                continue
            count += 1
            if num < 2:
                move_angle = angle
            else:
                move_angle = angle + np.pi / 2 / (num // 2) * (count // 2) * (-1) ** count
            move_pos = [target_pos[0] + dis * np.cos(move_angle), -1, target_pos[2] + dis * np.sin(move_angle)]
            self.move(move_pos, [uid], auth='normal', pos='replace', walkType='run', reachDist=3, maxDoTimes=1)
            threading.Thread(target=self.arrive_attack, args=(uid, move_pos, 6, True)).start()
            # time.sleep(2)
            # self.search_enemy_attack([uid],auth='normal',pos='tail')

    def add_ui_prompt(self, msg=''):
        '''
        左上角的ui提示
        '''
        self.show_ui_win('InfoWin', 1)
        cmd = 'cmd=add_ui_prompt`msg=%s'%msg
        self.client.send(cmd)
        threading.Thread(target=self.wait_close, args=('TaskWin', 5,)).start()

    def move_follow(self, objid_list='all', team_id=1, leader_objid=0, ):
        '''
        跟随
        '''
        print('move_follow')
        if type(objid_list) == int:
            objid_list = [objid_list]
        elif objid_list == 'all':
            objid_list = self.units.keys()

        ai = '<action name="MoveToPosAct" destObj="leader" walkType="run"  reachDist="6"/>'
        s = 'objid_list=%s`team_id=%d`auth=normal`group=group1`pos=replace`ai=%s' % \
            (list2str(objid_list), team_id, ai)
        self.make_action(s)

    def arrive_attack(self, uid, pos, dis=6, move_attack=True):
        '''
        围攻专用 因为目标点不一定可达所以2秒后强行下攻击指令
        '''
        print('arrive attack ', pos)
        for i in range(40):
            time.sleep(0.1)
            unit = self.states[uid]
            if unit['HEALTH'] <= 0:
                return
            cur_pos = unit['POSITION']
            if get_dis(pos, cur_pos, False) < dis:
                print('curpos:12',cur_pos)
                self.origin_ai([uid], move_attack=move_attack)
                return
        print('cur_pos:123',cur_pos)
        self.origin_ai([uid], move_attack=move_attack)

    def origin_ai(self, objid_list='all', team_id=-1, move_attack=True):
        '''
        使用原本ai替换当前ai
        '''
        print('origin_ai')
        if type(objid_list) == int:
            objid_list = [objid_list]
        elif objid_list == 'all':
            objid_list = self.units.keys()

        if move_attack:
            move_attack = '<action name="MoveToPosAct" destObj="target" walkType="run" reachDist="12"/>'
        else:
            move_attack = ''
        ai = '<check name="CheckTimeChk" interval="0"><check name="CanAttackTargetChk"> <action name="ShootAct"/> %s </check>' % (move_attack)
        ai += '<check name="CheckTimeChk" interval="0.2"> <action name="SearchEnemyAct"/>  <action name="SearchLeaderAct">'
        ai += '<action name="MoveToPosAct" destObj="leader" walkType="run"  reachDist="6"/> </action> <action name="PatrolAct"/> </check></check>'
        if team_id > -1:
            team_id_str = 'team_id=%d`' % team_id
        else:
            team_id_str = '' 
        s = 'objid_list=%s`%sauth=normal`group=group1`pos=replace`ai=%s' % (list2str(objid_list), team_id_str, ai)
        self.make_action(s)

        return s

    def register(self, key, val):
        self.refs[key] = val
    
    def getVal(self, key):
        return self.refs[key]
