import gym
from ppo2 import *
from ddpg import *
import gym_pendulum
from train_test_functions import *

#init agent state estimation
TRAIN_EPISODES = 50     # total number of episodes in each season
TEST_EPISODES = 10      # total number of episodes for testing
TRAIN_EPOCHS = 10   # training epochs in each season
GAMMA = 0.90   # reward discount
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
BATCH_SIZE = 100     # minimum batch size for updating PPO
MAX_BUFFER_SIZE = 200000     # maximum buffer capacity > TRAIN_EPISODES * 200
METHOD = 'clip'          # 'clip' or 'penalty'

##################
KL_TARGET = 0.01
LAM = 0.5
EPSILON = 0.2


#init agent pendulum
critic_lr_param = 0.0002
actor_lr_param = 0.0001
critic_lr = 0.002
actor_lr = 0.001
gamma = 0.99
tau = 0.005
upper_bound=3
lower_bound=-3

# #########################################################################################################################################################################
# #training
# for nr in range(438, 439):

#     # what variable we observe: angle, velocity or both
#     observed = "angle"

#     # noise level
#     R=1.0

#     #at what validation we can stop training
#     thres=-22

#     #make custom environment
#     env = gym.make("MyPendulum-v2")

#     #training
#     if observed == "both":
#         state_dim = (4,)
#     else:
#         state_dim = (3,)

#     action_dim = (3,)
#     action_bound = 0.5


#     # create an agent for state est and par est
#     agent = PPOAgent(state_dim, action_dim, BATCH_SIZE, MAX_BUFFER_SIZE,
#                          action_bound,
#                          LR_A, LR_C, GAMMA, LAM, EPSILON, KL_TARGET, METHOD)





#     agent_pendulum = actor_critic_pend(3, 1,
#                              critic_lr, actor_lr,
#                              gamma, tau,
#                              upper_bound, lower_bound,
#                              memory_capacity=50000,
#                              batch_size=64)


#     main_train(env, agent, agent_pendulum, observed, nr, R, thres)



# #########################################################################################################################################################################
#testing
# what variable we observe: angle, velocity or both
observed = "angle"


#make custom environment
env = gym.make("MyPendulum-v2")

   
if observed == "both":
    state_dim = (4,)
else:
    state_dim = (3,)

action_dim = (3,)
action_bound = 0.5


# create an agent for state est.
agent = PPOAgent(state_dim, action_dim, BATCH_SIZE, MAX_BUFFER_SIZE,
                         action_bound,
                         LR_A, LR_C, GAMMA, LAM, EPSILON, KL_TARGET, METHOD)


agent_pendulum = actor_critic_pend(3, 1,
                             critic_lr, actor_lr,
                             gamma, tau,
                             upper_bound, lower_bound,
                             memory_capacity=50000,
                             batch_size=64)


for i in range(438,439):
    R=1.0
    test(env, agent, agent_pendulum, observed,i,R)

