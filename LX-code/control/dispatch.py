class Dispatch(object):
    def __init__(self, model):
        self.model = model
        pass
    def act(self):
        #根据model计算各个据点派兵数量
        return self.model.calc_num()







