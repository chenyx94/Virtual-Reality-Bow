# FPS_env api

## set_env(self, SEVERIP='127.0.0.1', SERVERPORT=5123, client_DEBUG=False, env_DEBUG=False, speedup=1)
- env配置 在gym.make之后调用
- SEVERIP FPS服务器IP
- SERVERPORT FPS服务器端口
- client_DEBUG socket调试信息输出开关
- env_DEBUG env调试信息输出开关
- speedup 游戏加速时需匹配此项

## get_objid_list(self, name=1, pos=0)
- 请不要主动调用此函数
- 获取所有人id  名字和坐标可选

## get_pos(self, )
- 根据self.states获取所有人坐标
- return dict{id:pos}

## get_game_variable(self, )
- 请不要主动调用此函数
- 获取units所有状态 但不包括弹药数 敌我标记（暂时用team_id代替 -1为敌人 正数为我方）
- objid_list 为 'all' 或 list 值会保存在self.states中
- return dict{id:dict{key:value}}

## new_episode(self, save=1, replay_file=None, speedup=1, disablelog=0, scene_name='Simple')
- 开始新的一局 
- replay_file 回放的文件名 None时按时间自动生成
- 游戏的speedup请在IsLand.xml中设置
- disablelog 日志开关 0时崩溃速度会稍微快一些

## restart(self, sleep_time=10, port=-1)
- 重启游戏
- port -1时在相邻端口启动游戏 否则在port端口启动游戏

## playerai(self, )
- 主角进入ai模式

## ailog(self, objid)
- 输出某人的日志 reset不会改变该状态

## make_action(self, d)
- 做一个自定义动作 目前仅支持str类型

## add_obj(self, name, is_enemy, pos, leader_objid, team_id, dir=[0, 0, 0], model_name='DefaultAI', weapon='m4')
- 向地图指定位置加一个单位
- name:str
- is_enemy bool 标记是否为敌人，若为`True`，则`leader_objid`和`team_id`都应为-1
- pos: list of length 3 位置
- leader_objid: int 队长id，只对己方有效
- team_id: int 队伍编号，只对己方有效
- dir: list of length 3 `[angle,roll,pitch]`
- model_name: str 为AI模板不是人物模型
- weapon: str 武器类型

## add_obj_list(self, name, pos, leader_objid, team_id, width, num, is_enemy=False, dir=[0, 0, 0], model_name='DefaultAI')
- 在一个区域内随机添加一堆人 会重名

## check_pos(self, pos, objid=-1)
- 检查目标点对于某人来说是否可达 -1为随机一人

## move(self, destPos, objid_list='all', group='group1', auth='normal', pos='replace', walkType='walk', reachDist=6, maxDoTimes='', team_id=None, ):
- 强行移动不受其他因素干扰 挨打不还手用于撤退 队形变换
- reachDist 距离目标点的停止距离

## add_patrol_path(self, pos_list, objid, noteam=0, noleader=1)
- 添加一条巡逻路径

## add_map_mark(self, pos, marktype='mark', blinking_time=-1, lead_obj_id=0)
- 小地图添加标记
- type=mark|arrow|blinking|focus
- return mark_id

## remove_map_mark(self, mark_id)
- 根据id清除mark

## draw_pathline(self, pos_list)
- 绘制一条提示路径
- return path_id

## remove_pathline(self, path_id)
- 根据id清除path

## add_observer(self, pos, radius)
- 用于清除迷雾

## add_chat(self, msg, obj_id, close_time)
- 添加聊天记录 
- close_time 聊天框持续时间 

## select_obj(self, objid, is_select=1)
- 脚底高亮  重新选择、取消选择

## set_task(self, msg)
- 设置任务面板

## is_arrived(self, objid, target_pos, dis=-1)
- 是否到达目标点  dis为阈值
- return dis=-1时返回距离 否则返回距离是否小于dis

## create_team(self, leader_objid, member_objid_list, team_id)
- 编队  队员必须提前存在

## search_enemy_attack(self, objid_list='all', team_id=1, auth='normal', group='group1', pos='replace'):
- 搜索敌人并攻击
- 搁置!!!!!!!!!!!!!!

## can_attack_move(self, objid_list, destObjID='', destObj='', destPos='', team_id=None, auth='normal', group='group1', pos='replace', walkType='walk', reachDist=6)
- 移动时检查能否攻击 实际测试时经常反应慢一拍
- 移动优先级
- 1  destObjID 目标ID
- 2  destObj   target or leader
- 3  destPos   目标坐标

## attack(self, objid_list, auth='normal', pos='replace')
- 攻击  
- 若没有settargetact或searchenemyact指定target则无效

## set_target_objid(self, objid_list, targetObjID, auth='normal')
- 强行指定攻击目标 还需再调attack才会攻击

## get_enemy_nearby(self, team_id=1, mindis=30)
- 寻找附近的敌人
- return dict{id:dist}

## map_move(self, target_map_pos, objid_list='all', team_id=None, can_attack=True)
- 按照5x5坐标移动

## move_target(self, objid_list='all', target_id=0, team_id=1, walkType='run')
- 向某个人集中, 默认为第一队向主角跑步集中

## move_alert(self, team_id=1, capital_id=0, auth='normal', group='group1', walkType='run', dist=4, dist2=10, reachDist=1)
- 警戒移动  围一圈

## move_to_ahead(self, objid_list='all', team_id=1, capital_id=0, auth='normal', group='group1', walkType='run', angle=None, dist=4, reachDist=2)
- 挡住某人
- angle为阻挡方向 None时根据敌人自动选择

## attack_surround(self, target_objid=-1, team_id=1, capital_id=0, dis=15)
- 指定小队成员先包围再攻击
- 暂定为半圆  整圆太容易死
- 愚蠢的实现方法

## add_ui_prompt(self, msg='')
- 左上角的ui提示

## move_follow(self, objid_list='all', team_id=1, leader_objid=0, )
- 跟随

## arrive_attack
- 请不要主动调用该函数
- 围攻专用 因为目标点不一定可达所以2秒后强行下攻击指令

## origin_ai(self, objid_list='all', team_id=-1)
- 使用原本ai替换当前ai

## register(self, key, val)
- 注册一个变量

## getVal(self, key)
- 根据key获得一个变量的值