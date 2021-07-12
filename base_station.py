import math
import pdb
from Node import Node
import numpy as np
from constants import *


class BSs(Node):
    def __init__(self,width, noise): #基站的计算资源
        #self.busy = np.zeros((self.num_BS, self.num_channel), dtype=bool)  #信道的占用情况
        self.noise = noise  # per channel
        self.width = width  # per channel

    def set_channel():
        K = np.array([np.random.normal(4,1),np.random.normal(8,1),
                      np.random.normal(12,1),np.random.normal(16,1),
                      np.random.normal(20,1),np.random.normal(24,1)])
        return K

    def delay(self, UE, UEs, MECS, computation_consumption, action):
        I = self.interference(UE, UEs, MECS)
        x = UE.P_send * self.channel_gains[int(UE.number)][int(action[1])][int(action[2])] /(I + self.noise)
        rate = self.width * math.log((1 + x), 2)
        t = computation_consumption * action[0] / rate
        return t

        
'''
    def interference(self, UE, UEs, MECS, time):
        I = 0
        for i,oUE in enumerate(UEs):
            if i!= UE.number and oUE.new_task is not None:
                I += self.channel_gains[i][int(oUE.new_task.MEC)][int(oUE.new_task.action[2])] * oUE.P_send
        return I
            
    def raylifun(x,mu,sigma):
        pdf = x*np.exp(-((x - mu)**2)/(2*sigma**2))/sigma**2
        return pdf

    def set_channel(self):
        np.random.seed(0)
        a = np.random.random()
        b = np.random.random()
        for i in range(4):
            for j in range(2):
                self.channel_gains[i][j][0] = raylifun(a,i,j+1)
                self.channel_gains[i][j][1] = raylifun(b,i,j+1)
                
'''
