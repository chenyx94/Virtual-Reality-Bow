# coding=utf-8
from __future__ import division
import numpy as np
import math
from gym import spaces
from .. import utils  

from . import FPS_env as fc
import time
from .starcraft.Config import *
import copy

DISTANCE_FACTOR = 16
ENEMY = 1
MYSELF = 0


class doubleBattleEnv(fc.FPSEnv):
    def __init__(self):
        super(doubleBattleEnv, self).__init__()

        self.state = dict()
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        # self.state['myunits'] = {}
        self.myunits = {}
        self.state['game_over'] = False
        self.state['win'] = False
        self.units_id = []
        self.units_e_id = []
        self.units_dead_id = []
        self.current_my_units = {}
        self.current_enemy_units = {}
        self.episodes = 0
        self.episode_steps = 0
        self.init_my_units = {}
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.flag = True
        self.time1 = 0
        self.time2 = 0


    def _action_space(self):
        action_low = [-1.0, -math.pi/2, -1.0]
        action_high = [1.0, math.pi/2, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        # hit points, cooldown, ground range, is enemy, degree, distance (myself)
        # hit points, cooldown, ground range, is enemy (enemy)
        obs_low = np.zeros([1, 10])
        obs_high = (np.zeros([1, 10]) + 1) * 100
        return spaces.Box(np.array(obs_low), np.array(obs_high))


    def _reset(self):
        self.state = dict()
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        # self.state['myunits'] = {}
        self.myunits = {}
        self.state['game_over'] = False
        self.state['win'] = False
        self.units_id = []
        self.units_e_id = []
        self.units_dead_id = []
        self.current_my_units = {}
        self.current_enemy_units = {}
        self.episodes = 0
        self.episode_steps = 0
        self.init_my_units = {}
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.flag = True
        self.time1 = 0
        self.time2 = 0

        self.episodes += 1
        self.episode_steps = 0
        self.flag = 0
        self.new_episode()
        self.state['game_over'] = False
        self.state['win'] = False
        while len(self.states) == 0:
            time.sleep(0.1)          # 等待主角出现
        self.add_obj(name="敌人1", is_enemy=True, pos=[212.5, -1, 126.7], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人2", is_enemy=True, pos=[210.8, -1, 127.8], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人3", is_enemy=True, pos=[208.9, -1, 128.2], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人4", is_enemy=True, pos=[206.3, -1, 129.1], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人5", is_enemy=True, pos=[204.3, -1, 129.5], leader_objid=-1, team_id=-1)

        self.add_obj(name="队友1", is_enemy=False, pos=[150, -1, 113.5], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友2", is_enemy=False, pos=[151, -1, 114.0], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友3", is_enemy=False, pos=[153, -1, 113.6], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友4", is_enemy=False, pos=[155, -1, 113.0], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友5", is_enemy=False, pos=[157, -1, 112.3], leader_objid=-1, team_id=-1)
        time.sleep(Config.sleeptime)
        self.myunits={}
        self._make_feature()
        self.screen_my,self.screen_enemy = self._make_observation()
        self.obs = {}

        self.obs['screen_my'] = self.screen_my
        self.obs['screen_enemy'] = self.screen_enemy
        self.init_my_units = self.state['units_myself']
        unit_size = len(self.state['units_myself'])
        unit_size_e = len(self.state['units_enemy'])
        self.add_observer([-220, -1, 20], 2000) # disable the fog of war
        return self.obs, unit_size, unit_size_e



    def _action2cmd(self,action,uid,flag):
        ut = self.myunits[uid]
        x = ut['POSITION'][0]
        y = ut['POSITION'][2]
        print("before action: (x,y):({},{})".format(x,y))
        if action == 0:
            y -= 5
        elif action == 1:
            y += 5
        elif action == 2:
            x -= 5
        elif action == 3:
            x += 5
        elif action == 4:
            x += 5
            y -= 5
        elif action == 5:
            x += 5
            y += 5
        elif action == 6:
            x -= 5
            y += 5
        elif action == 7:
            x -= 5
            y -= 5
        elif action == 8:
            pass
        else:
            ut = {}
            if flag is 'myself':
                e_uid = self.units_e_id[action-9]
                ut = self.myunits[e_uid]
            else:
                e_uid = self.units_id[action-9]
                ut = self.myunits[e_uid]
            x = ut['POSITION'][0]
            y = ut['POSITION'][2]
        print("after action: (x,y):({},{})".format(x,y))



        return x,y
    def _make_commands(self, action, flag):
        cmds = []
        self.current_my_units = self.state['units_myself']
        self.current_enemy_units = self.state['units_enemy']
        if self.state is None or (len(action) == 0):
            return cmds
        if flag == 'myself':
            if len(action) is not len(self.units_id):
                return cmds
            # for uid, ut in self.state['units_myself'].items():
            for i in range(len(self.units_id)):
                uid = self.units_id[i]
                ut = self.myunits[uid]
                myself = ut
                if ut['HEALTH']<=0:
                    continue
                if action[i] > 8:
                    # Attack action
                    if myself is None:
                        return cmds
                    x2,y2 = self._action2cmd(action[i], uid,flag)
                    enemy_id, distance = utils.get_closest(x2, y2, self.state['units_enemy'])
                    cmds.append([0, uid, enemy_id])
                elif action[i] is not -1:
                    # Move action
                    if myself is None:
                        return cmds
                    x2,y2 = self._action2cmd(action[i], uid,flag)
                    cmds.append([1, uid, [x2, -1, y2]])

        else:
            # only has attack command
            if len(action) is not len(self.units_e_id):
                return cmds

            for i in range(len(self.units_e_id)):
                uid = self.units_e_id[i]
                ut = self.myunits[uid]
                myself = ut
                if ut['HEALTH']<=0:
                    continue
                if action[i] > 8:
                    # Attack action
                    if myself is None:
                        return cmds
                    x2,y2 = self._action2cmd(action[i], uid,flag)
                    enemy_id, distance = utils.get_closest(x2, y2, self.state['units_myself'])
                    cmds.append([0, uid, enemy_id])
                elif action[i] is not -1:
                    # Move action
                    if myself is None:
                        return cmds
                    x2,y2 = self._action2cmd(action[i], uid,flag)
                    cmds.append([1, uid, [x2, -1, y2]])


        # print "commands send!"
        return cmds

    def die_fast(self):
       # count_them = len(self.state['units_enemy'])
        cx_e, cy_e = utils.get_units_center(self.state['units_enemy'])
        cx, cy = utils.get_units_center(self.state['units_myself'])

        for uid, feats in self.state['units_myself'].items():
            self.move(objid_list=[uid], destPos=[cx_e, -1, cy_e], reachDist=3, walkType='run')
       # count_us = len(self.state['units_myself'])
        for uid, feats in self.state['units_enemy'].items():
            self.move(objid_list=[uid], destPos=[cx, -1, cy], reachDist=3, walkType='run')

        time.sleep(Config.sleeptime)
        self._make_feature()
        done = self.state['game_over']
        return done


    def _step(self, actions):
        self.episode_steps += 1
        action = actions[0].copy()
        action_e = actions[1].copy()
        
        commands = self._make_commands(action, 'myself')
        commands_e = self._make_commands(action_e, 'enemy')
        print('commands', commands)
        print('commands_e', commands_e)
        print("self.state['units_myself']:{}".format(self.state['units_myself']))
        print("self.state['units_enemy']:{}".format(self.state['units_enemy']))
        self.current_my_units = copy.deepcopy(self.state['units_myself'])
        self.current_enemy_units = copy.deepcopy(self.state['units_enemy'])
        self.time1 = time.time()
        print('time1', self.time1, 'time2', self.time2, "time gap", self.time1 - self.time2)     #第一次动作执行完到第二次动作开始
        for i in range(len(commands)):
            if commands[i][0] == 0:
                unit = self.states[commands[i][2]]
                self.states[commands[i][1]]['LAST_CMD']=[0, unit['POSITION'][0], unit['POSITION'][2]]
                self.set_target_objid(objid_list=[commands[i][1]], targetObjID=commands[i][2])
                self.attack(objid_list=[commands[i][1]], auth='normal', pos='replace')
            else:
                self.states[commands[i][1]]['LAST_CMD'] = [1, commands[i][2][0], commands[i][2][2]]
                self.move(objid_list=[commands[i][1]], destPos=commands[i][2], reachDist=3, walkType='run')
            self.states[commands[i][1]]['LAST_TIME'] = self.states[commands[i][1]]['TIME']
            self.states[commands[i][1]]['TIME'] = time.time()
            self.states[commands[i][1]]['LAST_POSITION_'] = self.states[commands[i][1]]['POSITION']
        for i in range(len(commands_e)):
            if commands_e[i][0] == 0:
                # print("wrong", commands_e[i][0])
                unit = self.states[commands_e[i][2]]
                self.states[commands_e[i][1]]['LAST_CMD'] = [0, unit['POSITION'][0], unit['POSITION'][2]]
                self.set_target_objid(objid_list=[commands_e[i][1]], targetObjID=commands_e[i][2])
                self.attack(objid_list=[commands_e[i][1]], auth='normal', pos='replace')
            else:
                self.states[commands_e[i][1]]['LAST_CMD'] = [1, commands_e[i][2][0], commands_e[i][2][2]]
                self.move(objid_list = [commands_e[i][1]], destPos=commands_e[i][2], reachDist=3, walkType='run')
            self.states[commands_e[i][1]]['LAST_TIME'] = self.states[commands_e[i][1]]['TIME']
            self.states[commands_e[i][1]]['TIME'] = time.time()
            self.states[commands_e[i][1]]['LAST_POSITION_'] = self.states[commands_e[i][1]]['POSITION']

        time.sleep(Config.sleeptime)
        self.time2 = time.time()
        self._make_feature()
        self.obs = {}
        self.screen_my, self.screen_enemy = self._make_observation()
        self.obs['screen_my'] = self.screen_my
        self.obs['screen_enemy'] = self.screen_enemy
        reward = self._compute_reward()
        print('reward', reward)
        done = self.state['game_over']
        unit_size = len(self.units_id)
        unit_size_e = len(self.units_e_id)
        print(unit_size, unit_size_e)
        return self.obs, reward, done, unit_size, unit_size_e




    def _make_feature(self):
        # init
        if len(self.myunits) == 0:
            for uid, ut in self.states.items():
                
                if ut['TEAM_ID'] > 0 and uid != 0 and ut['HEALTH'] > 0:
                    self.state['units_myself'][uid] = self.states[uid]
                    self.myunits[uid] = self.states[uid]
                    self.units_id.append(uid)

                elif ut['TEAM_ID'] <= 0 and uid != 0 and ut['HEALTH'] > 0:

                    self.state['units_enemy'][uid] = self.states[uid]
                    self.units_e_id.append(uid)
                    self.myunits[uid] = self.states[uid]
                else:
                    print("ut:{}".format(ut))

                    
            # print("for end!")
            self.units_id.sort()
            self.units_e_id.sort()

        # update
        else:
            pass
        
        for uid, ut in self.states.items():
            if ut['TEAM_ID'] > 0:
                if uid != 0 and ut['HEALTH'] > 0:
                    self.myunits[uid] = self.states[uid];
                    # print("self.state['units_myself'][uid] :{}".format(self.state['units_myself'][uid] ))
                elif uid!=0:
                    if uid not in self.units_dead_id:
                        self.units_dead_id.append(uid)
            else:
                if uid != 0 and ut['HEALTH'] > 0:
                    self.myunits[uid] = self.states[uid];
                elif uid!=0:
                    if uid not in self.units_dead_id:
                        self.units_dead_id.append(uid)
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        for uid, ut in self.myunits.items():
            print("ut:{}".format(ut))
            if ut['TEAM_ID'] > 0 and uid != 0 and ut['HEALTH'] > 0:
                self.state['units_myself'][uid] = ut              # alive
        
            elif uid != 0 and ut['HEALTH'] > 0:
                self.state['units_enemy'][uid] = ut           # alive
        if len(self.state['units_myself']) == 0 or len(self.state['units_enemy']) == 0:
            self.state['game_over'] = True
            if len(self.state['units_myself']) > 0:
                self.state['battle_won'] = True
            else:
                self.state['battle_won'] = False




    # I can change the parameter observation to the structured observation
    def _make_observation(self):
        from PIL import Image
        def array_to_img(array):
            array=array*255
            new_img=Image.fromarray(array.astype(np.uint8))
            return new_img
        # len(self.states) - 1 is because the main role is an observer.
        screen_my = np.zeros((86,86), dtype=int);
        screen_enemy = np.zeros((86,86), dtype=int);
        # observations = np.zeros([len(self.states) - 1, self.observation_space.shape[1]])  # [unit_size+enemy_size, 35]

        # print("self.states:{}".format(self.states))
        # raise("_make_observation test!!!!!!!!")
        #---------------------myself--------------------------------
        for ind in range(len(self.units_id)):
            uid = self.units_id[ind]
            ut = self.myunits[uid]
            if ut['HEALTH'] > 0:
                center_y = int(round(ut['POSITION'][0]/3.0))
                center_x = int(round(ut['POSITION'][2]/3.0))
                if center_x <= 0:
                    center_x = 1
                elif center_x >= 85:
                    center_x = 85
                if center_y <= 0:
                    center_y = 1
                elif center_y >= 85:
                    center_y = 85
                screen_my[center_x][center_y] = 2
        for ind in range(len(self.units_e_id)):
            uid = self.units_e_id[ind]
            ut = self.myunits[uid]
            if ut['HEALTH'] > 0:
                center_y = int(round(ut['POSITION'][0]/3.0))
                center_x = int(round(ut['POSITION'][2]/3.0))
                if center_x <= 0:
                    center_x = 1
                elif center_x >= 85:
                    center_x = 85
                if center_y <= 0:
                    center_y = 1
                elif center_y >= 85:
                    center_y = 85 
                screen_my[center_x][center_y] = (4 + ind)/1.0
        #---------------------enemy--------------------------------
        for ind in range(len(self.units_e_id)):
            uid = self.units_e_id[ind]
            ut = self.myunits[uid]
            if ut['HEALTH'] > 0:
                center_y = 85 - int(round(ut['POSITION'][0]/3.0))
                center_x = int(round(ut['POSITION'][2]/3.0))
                if center_x <= 0:
                    center_x = 1
                elif center_x >= 85:
                    center_x = 85
                if center_y <= 0:
                    center_y = 1
                elif center_y >= 85:
                    center_y = 85
                
                screen_enemy[center_x][center_y] = 2
        #----------------enemy-----------------------------
        for ind in range(len(self.units_id)):
            uid = self.units_id[ind]
            ut = self.myunits[uid]
            if ut['HEALTH'] > 0:
                center_y = 85 - int(round(ut['POSITION'][0]/3.0))
                center_x = int(round(ut['POSITION'][2]/3.0))
                if center_x <= 0:
                    center_x = 1
                elif center_x >= 85:
                    center_x = 85
                if center_y <= 0:
                    center_y = 1
                elif center_y >= 85:
                    center_y = 85
                
                screen_enemy[center_x][center_y] =   (4 + ind)/1.0
        #----------------------my wall---------------------------------
        for i in range(36,96):
            center_y = int(round(i/3.0))
            center_x = int(round(174/3.0))
            screen_my[center_x][center_y] = 1
            center_x = int(round(74/3.0))
            screen_my[center_x][center_y] = 1
        for i in range(160,220):
            center_y = int(round(i/3.0))
            center_x = int(round(174/3.0))
            screen_my[center_x][center_y] = 1
            center_x = int(round(74/3.0))
            screen_my[center_x][center_y] = 1
        #------------------------enemy wall-----------------------------------------
        for i in range(36,96):
            center_y = 85-int(round(i/3.0))
            center_x = int(round(174/3.0))
            screen_enemy[center_x][center_y] = 1
            center_x = int(round(74/3.0))
            screen_enemy[center_x][center_y] = 1
        for i in range(160,220):
            center_y = 85-int(round(i/3.0))
            center_x = int(round(174/3.0))
            screen_enemy[center_x][center_y] = 1
            center_x = int(round(74/3.0))
            screen_enemy[center_x][center_y] = 1

        screen_my_list = []
        for i in range(len(self.units_id)):
            uid = self.units_id[ind]
            ut = self.myunits[uid]

            center_y = int(round(ut['POSITION'][0]/3.0))
            center_x = int(round(ut['POSITION'][2]/3.0))
            if center_x <= 0:
                center_x = 1
            elif center_x >= 85:
                center_x = 85
            if center_y <= 0:
                center_y = 1
            elif center_y >= 85:
                center_y = 85
            
            tmp_screen = copy.deepcopy(screen_my)
            tmp_screen[center_x][center_y] = 20
            screen_my_list.append(tmp_screen)

        screen_enemy_list = []
        for ind in range(len(self.units_e_id)):
            uid = self.units_e_id[ind]
            ut = self.myunits[uid]
            center_y = 85 - int(round(ut['POSITION'][0]/3.0))
            center_x = int(round(ut['POSITION'][2]/3.0))
            if center_x <= 0:
                center_x = 1
            elif center_x >= 85:
                center_x = 85
            if center_y <= 0:
                center_y = 1
            elif center_y >= 85:
                center_y = 85
            
            tmp_screen = copy.deepcopy(screen_enemy)
            tmp_screen[center_x][center_y] = 20
            screen_enemy_list.append(tmp_screen)
        # file = open('./tmp/state.txt','w')
        # file.close()
        # file = open('./tmp/state.txt','a')     
        # for i in range(len(screen_my_list)):           
        #     for x in range(86):
        #         for y in range(86):

        #             if(screen_my_list[i][x][y]==20):
        #                 file.write("x:{},y:{}  ->".format(x,y))
        #                 file.write(str(screen_my_list[i][x][y]))
        #                 file.write("\n\n")
        # for i in range(len(screen_enemy_list)):           
        #     for x in range(86):
        #         for y in range(86):
        #             if(screen_enemy_list[i][x][y]==20):
        #                 file.write("x:{},y:{}  ->".format(x,y))
        #                 file.write(str(screen_enemy_list[i][x][y]))
        #                 file.write("\n\n")
        # file.close()

        # # img = array_to_img(screen_my_list[0])
        # # img.show()
        # raise("in the state")


        return screen_my_list,screen_enemy_list
 #    def _make_observation(self):
 #        # len(self.states) - 1 is because the main role is an observer.
 #        observations = np.zeros([len(self.states) - 1, self.observation_space.shape[1]])  # [unit_size+enemy_size, 35]
 #        if (len(self.states) <= 11):
 #            print("right")
 #        print("self.states:{}".format(self.states))
 #        raise("_make_observation test!!!!!!!!")
 #        count = 0
 #        for uid, ut in self.state['units_myself'].items():
 #            observations[count, 0] = uid
 #            observations[count, 1] = ut['HEALTH']/float(50)
 #            observations[count, 2] = ut['POSITION'][0]     #x
 #            observations[count, 3] = ut['POSITION'][2]     #y
 #            if 'LAST_CMD' not in ut.keys():
 #                observations[count, 6] = 0
 #                observations[count, 7] = 0
 #                observations[count, 8] = 0
 #                observations[count, 4] = 0
 #                observations[count, 5] = 0
 #            else:
 #                observations[count, 6] = ut['LAST_CMD'][1] / float(45)
 #                observations[count, 7] = ut['LAST_CMD'][2] / float(45)
 #                observations[count, 8] = ut['LAST_CMD'][0]
 #                observations[count, 4] = (ut['LAST_POSITION'][0] - ut['POSITION'][0]) / float(ut['TIME'] - ut['LAST_TIME'])
 #                observations[count, 5] = (ut['LAST_POSITION'][2] - ut['POSITION'][2]) / float(ut['TIME'] - ut['LAST_TIME'])
 #                #print(uid, ut['LAST_CMD'])
 # #           observations[count, 4] = unit.groundRange
 # #           observations[count, 5] = unit.groundATK
 # #           observations[count, 6] = unit.max_health/float(20)
 # #           observations[count, 7] = unit.orders[0].targetId / float(10)
 # #           observations[count, 11] = unit.type
 # #           observations[count, 12] = unit.velocityX
 # #           observations[count, 13] = unit.velocityY
 #            count += 1
 #        for uid,ut in self.state['units_enemy'].items():
 #            observations[count, 0] = uid
 #            observations[count, 1] = ut['HEALTH'] / float(50)
 #            observations[count, 2] = ut['POSITION'][0]  # x
 #            observations[count, 3] = ut['POSITION'][2]
 #            if 'LAST_CMD' not in ut.keys():
 #                observations[count, 6] = 0
 #                observations[count, 7] = 0
 #                observations[count, 8] = 0
 #                observations[count, 4] = 0
 #                observations[count, 5] = 0
 #            else:
 #                observations[count, 6] = ut['LAST_CMD'][1] / float(45)
 #                observations[count, 7] = ut['LAST_CMD'][2] / float(45)
 #                observations[count, 8] = ut['LAST_CMD'][0]
 #                observations[count, 4] = (ut['LAST_POSITION'][0] - ut['POSITION'][0]) / float(ut['TIME'] - ut['LAST_TIME'])
 #                observations[count, 5] = (ut['LAST_POSITION'][2] - ut['POSITION'][2]) / float(ut['TIME'] - ut['LAST_TIME'])
 #            count += 1
 #        return np.asarray(observations)
    # Health reward
    def _compute_reward(self):
        tmp_my = 0
        tmp_enemy = 0
        if len(self.current_my_units) == 0 or len(self.current_enemy_units) == 0:
            return None

        for uid, ut in self.current_my_units.items():      #action执行前
            if uid in self.init_my_units and uid not in self.state['units_myself']:
                tmp_my += ut['HEALTH']
            elif uid in self.state['units_myself']:
                tmp_my += ut['HEALTH'] - self.state['units_myself'][uid]['HEALTH']

        for uid, ut in self.current_enemy_units.items():
            if uid not in self.state['units_enemy']:
                tmp_enemy += ut['HEALTH']
            else:
                tmp_enemy += ut['HEALTH'] - self.state['units_enemy'][uid]['HEALTH']

        tmp_enemy /= len(self.current_enemy_units)
        tmp_my /= len(self.current_my_units)
        return tmp_enemy - tmp_my


