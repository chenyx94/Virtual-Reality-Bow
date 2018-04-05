# coding: utf-8

import math, time, random
import numpy as np
from gym import spaces
from .import FPS_env as fps

from ..utils import *

class DemoEnv2(fps.FPSEnv):
    '''
    TODO
    '''
    def __init__(self,):
        super(DemoEnv2, self).__init__()
        self.is_battled=False
        self.far_awaydespos=[-165.5,6.50,2.5]
        self.war_despos=[-225.4,-0.57,20.7]
        self.comradeteam=[]
        self.enemyteam=[]
        self.curcomradefeature={}
        self.curenemyfeature={}
    
    def transfer(self,action):
        actionmatrix=[]
        curlen=len(action)
        curoutput=3
        for i in range(curlen):
            if action[i][0]==0:
                updateaction=[]
                updateaction.append(0)
                curcomradeid=self.comradeteam[i]
                updateaction.append(curcomradeid)
                curpos=self.curcomradefeature[curcomradeid]['POSITION']
                curvector=[curpos[0],curpos[2]]
                for enemyid in self.enemyteam:
                    curdistance=[]
                    if  enemyid in self.curenemyfeature.keys():
                        curenemypos=self.curenemyfeature[enemyid]['POSITION']
                        curenemyvector=[curenemypos[0],curenemypos[2]]
                        curdisvector=np.array(curenemyvector)-np.array(curvector)
                        xdis=np.cos(action[i][1])*action[i][2]
                        ydis=np.sin(action[i][1])*action[i][2]
                        disvector=np.array([xdis,ydis])
                        dirdistance=np.vdot(curdisvector,disvector)/np.sqrt(np.vdot(disvector,disvector))
                        curdistance.append(dirdistance)
                    else:
                        curdistance.append(9999)
                shortindex=np.array(curdistance).argmin()
                shortid=self.enemyteam[shortindex]
                #print(self.enemyteam)
                #print(shortid)
                updateaction.append(shortid)
            else:
                updateaction=[]
                updateaction.append(1)
                curcomradeid=self.comradeteam[i]
                updateaction.append(curcomradeid)
                curpos=self.curcomradefeature[curcomradeid]['POSITION']
                xdis=np.cos(action[i][1])*action[i][2]
                ydis=np.sin(action[i][1])*action[i][2]
                destpos=[curpos[0]+xdis,-1,curpos[2]+ydis]
                updateaction.append(destpos)
            actionmatrix.append(updateaction.copy())
        return actionmatrix
    def search_enemy_nearby(objid_list,mindis=30):
        curlen=len(objid_list)



    def _step(self, action):
        curlen=len(action)
        for i in range(curlen):
            if action[i][0]==0:
                self.set_target_objid(objid_list=[action[i][1]],targetObjID=action[i][2])
            else:
                self.move(objid_list=[action[i][1]],destPos=[action[i][2]],reachDist=0)



    def _reset(self, ):
        self.new_episode()
        #self.playerai()
        #self.move(self.far_awaydespos,objid_list=[0],walkType='run',auth='top')
        self.add_obj(name="敌人1",pos=[-244.4,-0.19,66.6],leader_objid=-1,team_id=-1,is_enemy=True,model_name='CustomAI')
        self.add_obj(name="敌人2",pos=[-247.9,-0.63,66.4],leader_objid=-1,team_id=-1,is_enemy=True,model_name='CustomAI')
        self.add_obj(name="敌人3",pos=[-242.2,0.14,67.1],leader_objid=-1,team_id=-1,is_enemy=True,model_name='CustomAI')
        self.add_obj(name="敌人4",pos=[-250.9,-0.94,65.1],leader_objid=-1,team_id=-1,is_enemy=True,model_name='CustomAI')
        self.add_obj(name="敌人5",pos=[-238,0.81,67.2],leader_objid=-1,team_id=-1,is_enemy=True,model_name='CustomAI')
        self.add_obj(name="队友1",pos=[-248.8,-1.21,-15.2],leader_objid=-1,team_id=2,is_enemy=False,model_name='CustomAI')
        self.add_obj(name="队友2",pos=[-252.1,-0.98,-13.5],leader_objid=-1,team_id=3,is_enemy=False,model_name='CustomAI')
        self.add_obj(name="队友3",pos=[-253.9,-0.75,-11.3],leader_objid=-1,team_id=4,is_enemy=False,model_name='CustomAI')
        self.add_obj(name="队友4",pos=[-250.7,-1.05,-13.9],leader_objid=-1,team_id=5,is_enemy=False,model_name='CustomAI')
        self.add_obj(name="队友5",pos=[-255.5,-0.71,-12.1],leader_objid=-1,team_id=6,is_enemy=False,model_name='CustomAI')
        #time.sleep(6/self.speedup)
        statevariable=self.states

        for playerid,state in statevariable.items():
            if state['TEAM_ID']>0 and playerid!=0:
                self.comradeteam.append(playerid)
            elif playerid!=0:
                self.enemyteam.append(playerid)
        print(self.comradeteam)
        print(self.enemyteam)
        self.comradeteam.sort()
        self.enemyteam.sort()

        self.move(objid_list=self.enemyteam,destPos=self.war_despos)
        #time.sleep(5)
        self.move(objid_list=self.comradeteam,destPos=self.war_despos)

        #self.can_attack_move(objid_list=self.comradeteam,destPos=self.war_despos)
        #self.can_attack_move(objid_list=self.enemyteam,destPos=self.war_despos)


    def get_feature(self):
        vardict=self.states
        featuredict={}
        enemydict={}
        comradedict={}
        gameover=False
        win=-1
        for playerid,state in vardict.items():
            if playerid in self.comradeteam:
                comradedict[playerid]=state
            elif playerid!=0:
                enemydict[playerid]=state
        loseFlag=True
        winFlag=True
        for comradeid,state in comradedict.items():
            if state['HEALTH']>=1:
                loseFlag=False
                break    
        for enemyid,state in enemydict.items():
            if state['HEALTH']>=1:
                winFlag=False
                break
        if loseFlag:
            win=0
        if winFlag:
            win=1
        if win>=0:
            gameover=True
        self.curcomradefeature=comradedict
        self.curenemyfeature=enemydict
        self.comradeteam=list(comradedict.keys())
        self.enemyteam=list(enemydict.keys())
        featuredict['comrade']=comradedict
        featuredict['enemy']=enemydict
        featuredict['win']=win
        featuredict['gameover']=gameover
        featuredict['shootcd']=1.5
        featuredict['shootrange']=12
        return featuredict






