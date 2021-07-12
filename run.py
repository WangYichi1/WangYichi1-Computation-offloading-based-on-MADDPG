import numpy as np
import pdb
import matplotlib.pyplot as plt
from constants import *
from user_equipment import UserEquipment
from environment import UEsystem
from base_station import BSs
from mind import DDPG
from server import server
from myrandom import ran, ran01, result
import warnings

warnings.filterwarnings("ignore")
np.random.seed(0)

s_dim = 2+1+2*2+2 #改
a_dim = 3  #改
MEMORY_CAPACITY = 10000
UEnet = []

for i in range (4):
    UEnet.append(DDPG(a_dim, s_dim))
MEC1 = server(frequency, f_minportion) #每一个设备的最小份额应当不一样
MEC2 = server(frequency, f_minportion)
MECS=[MEC1,MEC2]

BSs = BSs(channel_gains, width, noise)


def random_create_UE(MECS, i):
    return UserEquipment(MECS, i, Prtask, ran(x1), ran(y1), ran(x2), ran(y2))

UEs = []
for i in range (4):
    UEs.append(random_create_UE(MECS, i))
   
time = 0    
env = UEsystem(MECS, UEs, time)


episode_history = np.zeros(4)
score_history = [[] for i in range (4)]
reward_history = [[] for i in range (4)]

env.reset(BSs)

#print(obs)
episode = [[] for i in range (4)]
for i in range (4):
    #pdb.set_trace()
    if UEs[i].new_task is not None:
        UEs[i].new_task.action = UEnet[i].choose_action(UEs[i].obs)
        UEs[i].new_task.action = UEs[i].start_work(UEs[i].new_task.action, BSs, UEs)
   
while time < total_time:
    #pdb.set_trace()
    BSs.set_channel()
    time += slot
    
    for i in range (4):
        if UEs[i].new_task is not None:
            state, reward, state_, action = UEs[i].finish(BSs, UEs[i].new_task.action, time, slot)
            episode_history[i] += 1
            score_history[i].append(reward)
            UEnet[i].store_transition(state, action, reward, state_)
            
            if UEs[i].new_task is not None:
                UEs[i].new_task.action = UEnet[i].choose_action(UEs[i].obs)
                UEs[i].new_task.action = UEs[i].start_work(UEs[i].new_task.action, BSs, UEs)
            
            UEnet[i].learn()

        else:
            UEs[i].set_tasks(time)
            if self.new_task is not None:
                self.get_state(BSs)
                UEs[i].new_task.action = UEnet[i].choose_action(UEs[i].obs)
                UEs[i].new_task.action = UEs[i].start_work(UEs[i].new_task.action, BSs, UEs)
            
            
            '''
            UEs[i].step(time, slot)
            if UEs[i].done:
                #pdb.set_trace()
                new_state, reward, done = UEs[i].finish(BSs, act[i], time, slot)
                #print(reward)
                episode_history[i] += 1
                score_history[i].append(reward)
                UEnet[i].store_transition(obs[i], act[i], reward, new_state)
                
                if UEnet[i].pointer > MEMORY_CAPACITY:
                    var *= .9995    # decay the action randomness
                    UEnet[i].learn()

                obs[i] = new_state
                if obs[i][0]:
                    act[i] = UEnet[i].choose_action(obs[i])
                    act[i] = UEs[i].start_work(act[i], BSs, UEs)
                else:
                    act[i] = False
            
        else:
            UEs[i].every_slot(time, slot)
            while UEs[i].buffer and UEs[i].task is None:
                obs[i] = UEs[i].get_state(BSs)
                act[i] = UEnet[i].choose_action(obs[i])
                UEs[i].start_work(act[i], BSs, UEs)
            '''
    

for i in range (4):
    for j in range (int(episode_history[i])):
        episode[i].append(j)


for i in range (4):
    for j in range (int(episode_history[i])):
        r = 0
        for k in range(j):
            r += score_history[i][k]
        r =r/(j+1)
        reward_history[i].append(r)
        
            
for i in range (4):
    plt.plot(episode[i],reward_history[i])
    plt.show()
            
  
'''        
    new_state, reward, done= env.step(act, act_index)
    UEnet[i].remember(obs[i], act, reward, new_state, int(done[i]))
        UEnet[i].learn()
        score_history[i].append(reward)
        obs[i] = new_state
    episode_history.append(episode)
    episode += 1

for i in range (10):
    plt.plot(episode_history,score_history[i])

plt.show()
'''









    
