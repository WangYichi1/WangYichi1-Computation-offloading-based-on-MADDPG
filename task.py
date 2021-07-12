import numpy as np
from math import *
import constants as cn

class Task():
    def __init__(self, data_size, delay_tolerant,
                 UE, arrival_timestamp = None):
        self.UE = UE
        self.data_size = data_size
        self.delay_tolerant = delay_tolerant
        self.computation_consumption = self.data_size * cn.cycle_number[self.UE.number]
        self.arrival_time = arrival_timestamp       
        self.fail_time = self.arrival_time + self.delay_tolerant
        self.finish_time =None
        self.local_finish = False
        self.reward = Reward(cn.fail_punish)
        self.f_assign = None
        self.MEC = None
        self.done = False
        self.action = None
        self.I = 0
    
    def get_data_size(self):
        return self.data_size
    
    def get_arrival_time(self):
        return self.arrival_time

    def set_offloading(self, MECS, UEs, now, action, apply_channels, ch_K, user_X, user_Y):
        self.MEC = MECS[int(action[1])]
        distance = sqrt(pow(user_X[self.UE.number]-MECS[int(action[1])].x,2)+pow(user_Y[self.UE.number]-MECS[int(action[1])].y,2))

        for i in (apply_channels[int(action[2])]):
            if i is not self.UE.number and UEs[i].MEC_flag is True:
                if i> self.UE.number:
                    deta_x = user_X[i]-MECS[int(UEs[i].new_task.action[1])].x
                    deta_y = user_Y[i]-MECS[int(UEs[i].new_task.action[1])].y
                    d = sqrt(pow(deta_x,2)+pow(deta_y,2))
                    self.I += UEs[i].P_send * ch_K[int(UEs[i].new_task.action[2])]*pow(1/d,2)
                else:
                    self.I = UEs[i].new_task.I
        #print("干扰",self.I)
        x = self.UE.P_send * ch_K[int(action[2])]* pow(1/distance,2)/(cn.noise+self.I)
        rate = cn.width * log((1 + x), 2)
        dtr = self.computation_consumption * action[0] / rate
        t_send = dtr + self.MEC.delay(self, action)
        #print("传送：",t_send)
        #print("速度：",rate)
        #print("MEC时间：",t_send-dtr)
        self.finish_time_send = t_send + now
        m = [t_send, dtr, self.finish_time_send]
        return m
        
    def set_local(self, now, action):
        self.arrival_time = now
        self.f_assign = cn.UE_frequency[self.UE.number]
        t_local = self.computation_consumption *(1- action[0]) / self.f_assign
        self.finish_time_local = now + t_local
        s = [t_local, self.finish_time_local]
        #print("本地：",t_local)
        return s
       
class Reward(object):
    def __init__(self, fail_punish):
        self.reward = 0
        self.fail_punish = fail_punish
    
    def task_failed(self):
        self.reward -= cn.w1 * exp(self.fail_punish)
        # raise Exception("only for debug")
        # print('任务失败后：reward is {}'.format(self.reward))
    
    def task_finish(self, time_delay):
        self.reward += cn.w2 * exp(-time_delay)
  
    def reset(self):
        self.reward = 0
        # print('重置后：reward is {}'.format(self.reward))
      
