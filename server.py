from user_equipment import UserEquipment
from task import Task
from constants import *
import numpy as np

class server(object):
    def __init__(self, i, mec_x, mec_y, frequency, f_minportion):
        self.num = i
        self.x = mec_x[i]
        self.y = mec_y[i]
        self.f = frequency[i]
        self.f_minportion = f_minportion[i]
        

    def delay(self, task, action):
        return task.computation_consumption * (1 - action[0]) / self.f_minportion


        

        
