import numpy as np
import pdb
import matplotlib.pyplot as plt
from constants import *
import random
from random import choice
from user_equipment import UserEquipment
from environment import UEsystem
from base_station import BSs
from smart_mind import DDPG
from server import server
from myrandom import ran, ran01, result
import warnings

warnings.filterwarnings("ignore")
np.random.seed(0)

s_dim = 2+1+6+1+1 #11
a_dim = 3   #1+2+6  
MEMORY_CAPACITY = 10000
epsilon = 0.9995
UEnet = []
MECS = []
for i in range (4):
    UEnet.append(DDPG(a_dim, s_dim))

for i in range (2):
    MECS.append(server(i, mec_x, mec_y, frequency, f_minportion))

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


#print(obs)
episode = [[] for i in range (4)]

apply_MEC1 = []
apply_MEC2 = []

apply_channel1 = []
apply_channel2 = []
apply_channel3 = []
apply_channel4 = []
apply_channel5 = []
apply_channel6 = []
apply_channels = [apply_channel1,apply_channel2,apply_channel3,apply_channel4,apply_channel5,apply_channel6]

def set_channel():
    K = np.array([np.random.normal(10,0.01),np.random.normal(8,0.01),
                  np.random.normal(12,0.01),np.random.normal(16,0.01),
                  np.random.normal(20,0.01),np.random.normal(24,0.01)])
    return K

def user_location_x():
    user_X = np.array([round(np.random.uniform(-1,1),2),round(np.random.uniform(9,11),2),
                       round(np.random.uniform(-11,-9),2),round(np.random.uniform(19,21),2),round(np.random.uniform(-21,-19),2)])
    return user_X

def user_location_y():
    user_Y = np.array([round(np.random.uniform(9,11),2),round(np.random.uniform(-1,1),2),
                       round(np.random.uniform(-1,1),2),round(np.random.uniform(9,11),2),round(np.random.uniform(9,11),2)])
    return user_Y

j=0
sum_show=0
while time < total_time:
    #pdb.set_trace()
    all_states = []
    all_actions = []
    sum_reward = 0
    all_states_next = []
    ch_K = set_channel()
    user_X = user_location_x()
    user_Y = user_location_y()
    #pdb.set_trace()
    env.reset(ch_K, user_X, user_Y)

    if j == 5:
        #print(sum_show)
        sum_show = 0
        j=j%5
    
    if j < 5:
        for i in range (4):
            if UEs[i].last_task is not None:
                sum_show = sum_show + UEs[i].last_reward
        j=j+1
    for i in range (4):
        if UEs[i].last_task is not None:
            #print("第",i,"个智能体的动作为：",UEs[i].last_action)
            all_states.append(UEs[i].last_obs)
            all_actions.append(UEs[i].last_action)
            sum_reward += UEs[i].last_reward
            all_states_next.append(UEs[i].obs)
            UEs[i].last_task = None
            UEs[i].last_obs = None
            UEs[i].last_action = None
            UEs[i].last_reward = None

    #print(sum_reward)
    if all_states:
        all_states = np.concatenate((all_states))
        all_actions = np.concatenate((all_actions))
        all_states_next = np.concatenate((all_states_next))
        for i in range (4):
            #print(i)
            UEnet[i].store_transition(all_states,all_actions,sum_reward,all_states_next)
            UEnet[i].learn(i)

    
        
    for i in range (4):
        UEs[i].new_task.action = UEnet[i].choose_action(UEs[i].obs,i)

        if UEs[i].new_task.data_size is not 0:
            if UEs[i].new_task.action[1]==0:
                apply_MEC1.append(i)
            else:
                apply_MEC2.append(i)

            if UEs[i].new_task.action[2]==0:
                apply_channel1.append(i)
            if UEs[i].new_task.action[2]==1:
                apply_channel2.append(i)
            if UEs[i].new_task.action[2]==2:
                apply_channel3.append(i)
            if UEs[i].new_task.action[2]==3:
                apply_channel4.append(i)
            if UEs[i].new_task.action[2]==4:
                apply_channel5.append(i)
            if UEs[i].new_task.action[2]==5:
                apply_channel6.append(i)

    
    if len(apply_MEC1) >= 2:
        sample_UE = random.sample(apply_MEC1, 2)
        UEs[sample_UE[0]].MEC_flag = True
        UEs[sample_UE[1]].MEC_flag = True
    if len(apply_MEC1)== 1:
        UEs[apply_MEC1[0]].MEC_flag = True

    if len(apply_MEC2) >= 2:
        sample_UE = random.sample(apply_MEC2, 2)
        UEs[sample_UE[0]].MEC_flag = True
        UEs[sample_UE[1]].MEC_flag = True
    if len(apply_MEC2)== 1:
        UEs[apply_MEC2[0]].MEC_flag = True        

    for i in range(4):
        if UEs[i].new_task.data_size is not 0:
            UEs[i].start_work(UEs[i].new_task.action, UEs, time, apply_channels, ch_K, user_X, user_Y)
            
    time += slot
    
    for i in range (4):
        if UEs[i].new_task.data_size is not 0:
            UEs[i].finish(UEs[i].new_task.action, time, slot)
        else:
            UEs[i].new_task.reward.reward = 0
        episode_history[i] += 1
        score_history[i].append(UEs[i].last_reward)

    apply_MEC1.clear()
    apply_MEC2.clear()
    apply_channel1.clear()
    apply_channel2.clear()
    apply_channel3.clear()
    apply_channel4.clear()    
    apply_channel5.clear()
    apply_channel6.clear()    
            
'''            
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
    

'''
////////////////////////////////////////////////////
需要的部分
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
            
//////////////////////////////////////////////////////////       
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









    
