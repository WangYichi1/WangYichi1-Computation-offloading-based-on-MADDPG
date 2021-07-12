import math
import numpy as np

# Data size scales
BYTE = 8
KB = 1024*BYTE
MB = 1024*KB
GB = 1024*MB
TB = 1024*GB
PB = 1024*TB

# CPU clock frequency scales
KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3


# Time scales
slot = 1 # seconds？？？？ 1s
#action_slot = 5e-2  # seconds ---- Handle this carefully!!!
time_total = 300  # seconds   total simulation time 

width = 20*MHZ  # MHz 带宽
noise = 1.5e-4  # W 噪声

# MECS
frequency = np.array([300,100])*GHZ  # GHz CPU转数
f_minportion = np.array([15,5])*GHZ  # GHz  # not implemented yet 分配的一份额的转数

mec_x = [10,-10]
mec_y = [10,10]

UE_frequency = np.array([20, 30, 40, 50])*GHZ  # GHz Unified distributed
part_frequency = 0.4*GHZ

full_Battery = np.array([1000, 4000])*3.7*3.6
# Joule -- mAh*Volt*3.6 Unified distributed

# Pr(charge_begin) = max(x1 * battery/FullBattery +y1, 0)
# Pr(charge_end) = max(x2 * battery/FullBattery +y2, 0)
x1 = np.array([-0.001, -0.0005])*10
y1 = np.array([3e-4,8e-4])*10
x2 = np.array([0.001, 0.002])
y2 = np.array([-3e-4,-8e-4])



# Tasks
Prtask = 80 * slot  # Probability of task coming at any slot
'''
data_size = np.array([[55,53,51,45,0],
                     [85,84,40,20,0],
                     [120,100,95,90,0],
                     [35,34,33,32,0]])*KB  # kB Unified distributed
'''
data_size = np.array([[55,55,55,55,55],
                     [85,85,85,85,85],
                     [120,120,120,120,120],
                     [35,35,35,35,35]])*KB

cycle_number = np.array([7.375e2,6.345e2,5.678e2,8.567e2,4.345e2])
# CPU_cycle Unified distributed

# training
episode = 2000  # maximum episode
# for the loss function:
fail_punish = -1  # per task
delay_tolerant = 3  # seconds

w1 = 300
w2 = 300
total_time = 10000000
min_level = 1e-3
P_send = 0.5
