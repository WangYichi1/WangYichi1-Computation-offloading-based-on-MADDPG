import constants as cn
import numpy as np
import math
from user_equipment import UserEquipment
from task import Task

class UEsystem(object):
    def __init__(self, MECS, UEs, time):
        self.UEs = UEs
        self.time = time
        self.MECS = MECS
        self.slot = cn.slot
        #self.action_c = int(cn.action_slot/self.slot)
        #self.initialize()
        
    def initialize(self):
        for UE in self.UEs:
            UE.set_tasks(self.time)  #可能没有新的任务产生new_task为None
    
    def reset(self, ch_K, user_X, user_Y):
        self.initialize()
        for UE in self.UEs:
            #if UE.new_task is not None:
            UE.get_state(ch_K, user_X, user_Y)



        
            

        
        
