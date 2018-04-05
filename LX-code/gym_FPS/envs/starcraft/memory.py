
class Memory(object):
    def __init__(self, s, s1 ,a, r, s_,s1_, unit_size, unit_size_):
        self.s = s
        self.s1 = s1
        self.a = a
        self.r = r
        self.s_ = s_
        self.s1_ = s1_
        self.unit_size = unit_size
        self.unit_size_ = unit_size_

    def getS(self):
        return self.s

    def getS1(self):
        return self.s1

    def getA(self):
        return self.a

    def getR(self):
        return self.r

    def getS_(self):
        return self.s_

    def getS1_(self):
        return self.s1_

    def getUnit_size(self):
        return self.unit_size

    def getUnit_size_(self):
        return self.unit_size_

