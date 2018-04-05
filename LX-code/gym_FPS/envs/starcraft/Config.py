import configparser

cf = configparser.ConfigParser()
cf.read('config.dat', encoding='utf-8')

class Config(object):

    serverip = cf.get('fps', 'serverip')
    serverport = cf.getint('fps', 'serverport')
    speed = cf.getfloat('fps', 'speed')
    frame_skip = cf.getint('fps', 'frame_skip')
    sleeptime = cf.getfloat('fps', 'sleeptime')
    max_dist = cf.getfloat('fps', 'max_dist')
    try:
        game_dir = cf.get('fps', 'game_dir')
        wait_for_game_start = cf.getfloat('fps', 'wait_for_game_start')
    except:
        pass


    hidden_size = cf.getint('model', 'hidden_size')
    episode = cf.getint('model', 'episode')
    reset_time = cf.getint('model', 'reset_time')
    load = cf.getint('model', 'load')
    device = cf.get('model', 'device')
    lr_a = cf.getfloat('model', 'lr_a')
    lr_c = cf.getfloat('model', 'lr_c')
    gamma = cf.getfloat('model', 'gamma')
    std = cf.getfloat('model', 'std')
    max_to_keep = cf.getint('model', 'max_to_keep')
    memory_capacity = cf.getint('model', 'memory_capacity')
    batch_size = cf.getint('model', 'batch_size')
    enemy_num = cf.getint('model', 'enemy_num')
    replay_start_size = cf.getint('model', 'replay_start_size')

    episode_to_save = cf.getint('utils', 'episode_to_save')
    episode_to_reset_win = cf.getint('utils', 'episode_to_reset_win')
    reward_dir = cf.get('utils', 'reward_dir')
    win_dir = cf.get('utils', 'win_dir')
    loss_dir = cf.get('utils', 'loss_dir')
    model_dir = cf.get('utils', 'model_dir')
    summary_dir = cf.get('utils', 'summary_dir')
    memory_dir = cf.get('utils', 'memory_dir')
    net_output_dir = cf.get('utils', 'net_outputdir')


if __name__ == '__main__':
    config = Config()
