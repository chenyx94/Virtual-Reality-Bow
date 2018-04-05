# ccntfps

## 11.25
- demo_env3 提供了其他交互设备的接口 样例见demo_env3_example.py 数据处理样例见server.py
- 由于top级指令存在move后无法攻击的bug 实在无法找到简易的ai指令完成围攻动作 先采用的替代方案添加线程检测unit是否到达目的地

### TODO
- 几个阵型的距离参数尚未调整 阵型较难看
- 包围目前只是简单的走直线 是否需要调整？？？


## 0110
- demo_env用于5x5大地图移动 demo_env2为肖祥写的5x5对战环境 demo_env3为最小粒度的step环境
- vr_version_fps_show_script 为演示流程 network为数据中转，sendeye模拟眼动脑机手势输入 


## 0124
- 支持自动开启FPS.exe 路径在config.dat中game_dir配置 同时游戏本身的配置文件将不会生效 请复制到InitConfig中
- 支持同时开启多个FPS运行  启动端口请在config.dat port配置 在第一个set_env后追加一次restart(port=CONFIG.port)即可
- 修复starcraft中移动方向计算的错误
- 补充速度相关特征

## 存在的问题
- 多人同时在不同端口运行时有时会收到一些意料外的指令？？？