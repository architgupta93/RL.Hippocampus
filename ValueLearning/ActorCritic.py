class Actor(object):
    def __init__(self, actions):
        pass
    
    def getAction(self):
        raise NotImplementedError

class Critic(object):
    def __init__(self):
        pass
    
    def getValue(self):
        raise NotImplementedError