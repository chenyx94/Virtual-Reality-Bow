# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tkinter.messagebox 
import time
import signal
import numpy as np

class MyNPC(object):
    
    def __init__(self, env):
        self.env = env 
        self.ours = []
        
    def is_alive(self, plist):
#        print('test_alive',plist,self.ours)
        for bro in self.ours:
            if bro in plist:
                return bro
        return None
    
    def arrive(self, tar):
        
        players=self.env.units
        searchplayer=self.env.get_pos()
        #print(players)
        for bro in self.ours:
            if bro in players and self.small(searchplayer[bro], tar):
                print('ARRIVEDDDD',tar)
                return True
        return False
    
    def small(self, pos1, pos2, dis=10):
        pos1=[pos1[0],pos1[2]]
        pos2=[pos2[0],pos2[2]]
        distance=np.array(pos1)-np.array(pos2)
        gap=np.mat(distance)*np.mat(distance).T
        gap_double=np.sqrt(gap[0][0])
        if gap_double<dis:
            return True
        return False
    
class Scout(MyNPC):
    
    def __init__(self, me, env, mode, leader, targets, follows):
        super(self.__class__, self).__init__(env)
        self.mode = mode
        self.leader = leader
        self.ours.append(me)
        self.tars = targets
        self.follows = follows
    
    def wait_for_leader(self):
        #print('chk',self.leader,self.env.getVal("wait_"+self.leader))
        if self.env.getVal("wait_"+self.leader) > 0:
            return True
        return False
    
    def run(self):
#        dead = False
        for i in range(len(self.tars)):
            tar = self.tars[i]
            self.env.move(tar, objid_list=self.ours,walkType='run')
        '''
        count = 0
        while count < 100:
            time.sleep(3)
            count+=1
            if self.wait_for_leader():
                print("get leader!")
                break
        
        last_pos = 0
        while True:
            pos = self.env.getVal("wait_"+self.leader)
            
            #we win
            players=self.env.units
            if self.is_alive(players) is None:
                break
            print('the current leader pos is %d' % pos)
            if pos >= len(self.follows):
                break
            if last_pos != pos:
                if pos==len(self.follows)-3 and self.leader=='team2':
                    if self.env.getVal('sys_team1') and self.env.getVal('sys_team2'):
                        self.env.can_attack_move(objid_list=self.ours,destPos=self.follows[pos])
                        last_pos = pos
                elif pos==len(self.follows)-1 and self.leader=='team2':
                    if self.env.getVal('ready_team1') and self.env.getVal('ready_team2'):
                        self.env.can_attack_move(objid_list=self.ours,destPos=self.follows[pos])
                        last_pos = pos
                else:
                    self.env.can_attack_move(objid_list=self.ours,destPos=self.follows[pos])
                    last_pos = pos             
            time.sleep(2)
        '''
#        last_tar2 = None
#        while True:
#            tar2 = self.env.getVal(self.leader)
#            if last_tar2 != tar2:
#                self.env.move(tar2, objid_list=self.ours,walkType='run')
#            time.sleep(15)
        
class TeamWork(MyNPC):
    
    def __init__(self, label, env, ours, targets):
        super(self.__class__, self).__init__(env)
        self.tars = targets
        self.label = label
        self.flag=True
        for our in ours:
            self.ours.append(our)
            
    def wait_for_arrive(self, tar):
        count = 0
        last_pos = None
#        if tar[0]==-100.6:
#            time.sleep(15)
#            return False
        while True:
            time.sleep(5)
            count+=1
            try:
                playerlist=self.env.units
                searchplayer=self.env.get_pos()
                bro = self.is_alive(playerlist)
                print(bro)
                if bro == None:
                    print('DDDDEADDDD!',self.label)
                    self.env.register('ready_'+self.label, True)
                    return True         
#                print('test7',self.label)
                pos = searchplayer[bro]
#                print('test8',self.label,pos)
    #            self.env.register(self.label, pos)
                #print(searchplayer)
                #print('111111111111111111')
                #print(last_pos)
                #print(pos)
                
                if  (last_pos is not None) and (pos is not None) and self.small(last_pos, pos) :
                    count+=1
#                print('test9',self.label)
                #print('3333333333333333333333')
                if tar[0]!=-100.6:
                    if count >= 50 and tar[0]!=-58.6 or \
                        self.env.getVal("clear1") and tar[0]==-154.1:
                        print('CHANGE TO NEXT POSITION',self.label)
                        return False
                    else:
                        last_pos = pos
                
                print(self.label,'is arrived???',tar,pos,isinstance(pos, str))
                #if isinstance(pos, str):
                    #print('xxx1',self.label)
                    #return False                 
                if self.arrive(tar):
                    return False
                
                try:
                    if tar[0]==-100.6 and not isinstance(pos, str):
                        if abs(tar[0]-pos[0])+abs(tar[2]-pos[2])<20:
                            print('!!!',abs(tar[0]-pos[0])+abs(tar[2]-pos[2]))
                            return False
#                    if isinstance(pos, str):
#                        print('xxx2',self.label)
          
                except:
                    print('problem!')
                    pass
            except:
                print('kakaka!')
                if tar[0]==-120.6:
                    return False
                pass
        return False
    
    def run(self):
        for i in range(len(self.tars)):
            print('runit!',self.label,i)
            tar = self.tars[i]
            if i == len(self.tars)-1:
                print('final test',self.label,self.env.getVal('ready_team1'),self.env.getVal('ready_team2'))
                while not self.env.getVal('ready_team1') or \
                not self.env.getVal("ready_team2"):
                    self.env.set_task("分队和主队集合")
                    if self.env.getVal("ready_team1") and not self.env.getVal("ready_team2"):
                        if self.label=='team2':
                            team2seg2="分队正在朝目标点集合"
                            self.env.add_chat(msg=team2seg2,obj_id=self.ours[0])
                    elif not self.env.getVal("ready_team1") and self.env.getVal("ready_team2"):
                        if self.label=='team2':
                            team2seg2="分队集结完毕，正在等待主队"
                            self.env.add_chat(msg=team2seg2,obj_id=self.ours[0])
                    else:
                        if self.label=='team2':
                            team2seg2="分队正在朝目标点集合"
                            self.env.add_chat(msg=team2seg2,obj_id=self.ours[0])
                    time.sleep(5)
                team2task2='攻击第二据点'
                self.env.set_task(team2task2)
                team2msg2="本小队将攻击第二据点"
                players=self.env.units
                prerole=self.is_alive(players)
                self.env.add_chat(msg=team2msg2,obj_id=prerole)
            if i==len(self.tars)-3:
                print('sys_team1 is',self.env.getVal('sys_team1'))
                print('*********************************************************')
                count=0
                while not self.env.getVal('sys_team1') or not self.env.getVal('sys_team2'):
                    self.env.set_task('分队与主队合击据点')
                    #self.hold_position()
                    if not self.env.getVal('sys_team1'):
                        team2seg3="分队正在等待主队的合击指令"
                        if count%10==0:
                            players=self.env.units
                            role=self.is_alive(players)
                            self.env.add_chat(msg=team2seg3,obj_id=role)
                            count=0
                        count+=1
                    time.sleep(1)
                self.env.set_task("两路夹击")
                #time.sleep(1)
            premark=self.env.add_map_mark(pos=self.tars[i])   
            self.env.can_attack_move(objid_list=self.ours,destPos=tar)
            all_dead = self.wait_for_arrive(tar)
            if all_dead:
                print('all dead',self.label)
                self.env.remove_map_mark(premark)
                break
            else:
                self.env.remove_map_mark(premark)
                self.env.register("wait_"+self.label,i+1)
                #print(self.label,'dbgx1',i,len(self.tars))
                players=self.env.units
                role=self.is_alive(players)
                if i == len(self.tars)-2: 
                    self.env.register('ready_'+self.label, True)
                if i == len(self.tars)-4:
                    print('we want to asychorize')
                    self.env.register('sys_'+self.label,True)
                    if self.label=='team2':
                        team2seg1='分队在行进过程中发现哨塔，直接采取攻击方式'
                        self.env.add_chat(msg=team2seg1,obj_id=role)
                if i == len(self.tars)-3:
                    self.lastfocus=self.env.add_map_mark(pos=self.tars[-1],marktype='focus')
                    team2task='主队与分队集合'
                    self.env.set_task(team2task)
                    team1msg='已经消灭第一据点敌人，将前往下一地点集合'
                    self.env.add_chat(msg=team1msg,obj_id=role)
    def hold_position(self):
        curpos=self.env.get_pos()
        for id in self.ours:
            if id in self.env.units:
                self.env.move(destPos=curpos[id],objid_list=[id])
                    
class Moniter(object):
    
    def __init__(self, env):
        self.env = env
    
    def check(self, checks, code):
        if self.env.getVal(code)==True:
            return True
        
        playerdict=self.env.units
        playerlist=[]
        for _,player_name in playerdict.items():
            if  not isinstance(player_name, str):
                   return False
            playerlist.append(player_name)
        #print('ck',checks)
        #print('pl',playerlist)
        for ck in checks:
            if ck in playerlist:
                return False
        
        self.env.register(code,True)
        return True
    
    def run(self):
#        return None
        print('we get monitor')
        do_it = False
        while True:
            try:
                #print("TASK:",self.env.getVal("clear1"),self.env.getVal("clear2"))
                clear1 = self.check(["驻守1","驻守2","驻守3","驻守4","驻守5","驻守6","驻守7","驻守8","驻守9"], "clear1")
                if clear1:
                    clear2 = self.check(["驻守10","驻守11","驻守12","驻守13","驻守14","驻守15","驻守16","驻守17","站岗3"],"clear2")
                    if clear2:
                        do_it = True
                        break
            except:
                print('TASK PASS')
                pass
            time.sleep(3)
        print('why no result?')
        if do_it ==True:
            print('what is matter?')
            self.env.add_ui_prompt("战斗胜利，游戏已经结束。")
            
#        self.env.search_enemy_attack()