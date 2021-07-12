from constants import *
import numpy as np
import logging
from Node import Node
from task import Task

class UserEquipment(Node):
    def __init__(self, MECS, i, Prtask, x1, y1, x2, y2):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.MECS = MECS
        self.Prtask = Prtask
        self.last_task = None
        self.last_obs = None
        self.last_action = None
        self.last_reward = None
        self.s = None
        self.m = None
        self.number = i
        self.P_send = P_send
        self.new_task = None
        self.f = UE_frequency[i]
        self.obs = None
        self.MEC_flag = False
        self.task_index = 0
        
        
    def set_tasks(self, time):
            self.new_task = self.random_create_task(time)

        
    def start_work(self, action, UEs, time, apply_channels, ch_K, user_X, user_Y):  #只针对新的任务
        self.new_task.reward.reset()
        self.new_task.action = action
        if self.MEC_flag is True:
            s = self.new_task.set_local(time, action)
            m = self.new_task.set_offloading(self.MECS, UEs, time, action, apply_channels, ch_K, user_X, user_Y)
            if s[-1] > m[-1]:
                self.new_task.finish_time = s[-1]
            else:
                self.new_task.finish_time = m[-1]
        else:
            self.new_task.reward.task_failed()
            self.new_task.done = True
            
    def finish(self, action, time, slot):
        if not self.new_task.done:
            if self.new_task.finish_time < time:
                self.new_task.reward.task_finish(self.new_task.finish_time)
                
            else:
                self.new_task.reward.task_failed()
            self.new_task.done = True

        self.last_task = self.new_task
        self.last_obs = self.obs
        self.last_action = action
        self.last_reward = self.new_task.reward.reward

        self.new_task = None
        self.obs = None
        self.MEC_flag = False
        self.s = None
        self.m = None
    
    def get_state(self, ch_K, user_X, user_Y):
        x = self.task2np(self.new_task)
        self.obs = np.concatenate((x.reshape(-1), [self.f], ch_K.reshape(-1), [user_X[self.number]], [user_Y[self.number]]))
        
    def task2np(self, x):
        m = [x.data_size, x.computation_consumption]#或者只输入一个试一下？
        return np.array(m)


    def random_create_task(self, time):
        self.task_index =(self.task_index + 1)%5
        return Task(data_size[self.number][self.task_index], delay_tolerant, self, arrival_timestamp = time)
            

