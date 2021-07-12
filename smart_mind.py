import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice
import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import pdb

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.1     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 100
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(s_dim,30)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc1.bias.data.normal_(0,0.1)
        self.out = nn.Linear(30,a_dim)
        self.out.weight.data.normal_(0,0.1) # initialization
        self.out.bias.data.normal_(0,0.1)
        self.f=[]
    def forward(self,x):
        #pdb.set_trace()
        #print("状态：",x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.out(x)
        x = F.tanh(x)
        x = x*x   #X1是0-1之间的数
        return x
    
class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(s_dim*4,30)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim*4,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value

'''
m = CNet(2,3)
print(m.state_dict())#.keys())
'''


class DDPG(object):
    def __init__(self, a_dim, s_dim,):
        self.a_dim, self.s_dim = a_dim, s_dim,
        self.memory = np.zeros((MEMORY_CAPACITY, (s_dim *2 + a_dim)*4 + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.epsilon = 0.9995

    def choose_action(self, s, i):
        self.Actor_eval.eval()
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        #action = self.Actor_eval(s)[0].detach()
        if np.random.random() > self.epsilon: 
            action = self.Actor_eval(s)[0].detach()
            #print(action)
            if action[1]<0.5:
                action[1]=0
            else:
                action[1]=1
                
            if action[2]<0.5:
                action[2]=0
            else:
                action[2]=1
            #print("第",i,"个智能体动作为：",action)
        else:
            action = torch.Tensor([np.random.random(),choice([0,1]),choice([0,1])])
            self.epsilon = self.epsilon*0.9995
        
        return action

    def learn(self,n):
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement  self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        if self.pointer >= BATCH_SIZE:
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
            indices = np.random.choice(MEMORY_CAPACITY, size=self.pointer)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim*4])
        ba = torch.FloatTensor(bt[:, self.s_dim*4: self.s_dim*4 + self.a_dim*4])
        br = torch.FloatTensor(bt[:, -self.s_dim*4 - 1: -self.s_dim*4])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim*4:])

        #Actor网络更新如下：
        array_bs = bs.reshape(int(len(bs.reshape(-1))/11),11)  #将bs切片
        a = []
        for i in range(4):
            a.append(self.Actor_eval(array_bs[i]).tolist())  
        a = torch.FloatTensor(a)
        a = a.reshape(-1)
        q = self.Critic_eval(bs,a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward(retain_graph=True)
        self.atrain.step()

        #打印Actor网络梯度
        for parms in self.Actor_eval.parameters():
            print('-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)

        #Critic网络更新如下
        array_bs_ = bs_.reshape(int(len(bs_.reshape(-1))/11),11)
        a_ = []
        for i in range(4):
            a_.append(self.Actor_target(array_bs_[i]).tolist())  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        a_ = torch.FloatTensor(a_)
        a_ = a_.reshape(-1)
        q_ = self.Critic_target(bs_,a_)
        q_target = br+GAMMA*q_  # q_target = 负的
        q_v = self.Critic_eval(bs,ba)
        td_error = self.loss_td(q_target,q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()



    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


