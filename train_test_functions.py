
import pickle
import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
from tensorflow.keras import layers, Model
from tensorflow import keras
from scipy import signal
import matplotlib.pyplot as plt
from ddpg import *
from ppo2 import *
import gym_pendulum
from helpers import *

############################


############### 

TRAIN_EPOCHS = 60      # training epochs in each season

################

# training loop
def main_train(env, agent, agent_pendulum, observed, nr, R, thres):
    print(nr, R, observed)

    path = './'

    if observed == "angle":
        env.set_angle("angle")
    if observed == "velocity":
        env.set_angle("velocity")
    if observed == "both":
        env.set_angle("both")
    end_training=False

    # training
    max_episodes = 150
    total_steps = 0
    best_score = -np.inf
    best_score_pend = -np.inf
    best_diff = np.inf
    ep_reward_list = []
    ep_res_swing=[]
    ep_res_par=[]

    #init noise ddpg
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    for ep in range(max_episodes):

        #set initial system noise parameters for t=0
        angle_noise = np.random.normal(0,  0.01**2)
        vel_noise = np.random.normal(0,  0.01**2)
        env.set_angle_noise(angle_noise)
        env.set_vel_noise(vel_noise)

        #reset true state and est. state for each episode
        true_state = env.reset()
        state = env.reset_est()

        #init start state of the pendulum
        state_pend = np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

        #get first torque based on the pendulum state
        state_pend = tf.squeeze(tf.expand_dims(tf.convert_to_tensor(state_pend), 0))
        action_swing =agent_pendulum.policy(state_pend, ou_noise)

        ep_reward = 0
        ep_reward_pend = 0
        t = 0

        if observed == "angle":
            y_th = get_next_meas_angle(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R) #measurement 
            state_for_est = np.array([state[0], state[1], y_th], dtype=np.float32) #input state estimation
            y_th_est = state[0]

        elif observed == "velocity":
            y_th = get_next_meas_vel(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R) #measurement 
            state_for_est = np.array([state[0], state[1], y_th], dtype=np.float32) #input state estimation
            y_th_est = state[1] 

        else:
            newth, newthdot = get_next_meas(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R)
            y_th=[newth,newthdot] #measurement 
            y_th_est = state
            state_for_est = np.array([state[0], state[1], y_th[0], y_th[1]], dtype=np.float32) #input state estimation

            
        print("episode: ", ep)

        while True:

                if end_training==False:
                

                    #get action based on estimation
                    action = agent.policy(state_for_est, y_th, y_th_est, observed)

                    #total action: estimated state and control input
                    total_action = np.array([action[0][0], action[0][1], action_swing, y_th, action[0][2]])

                    #input total action to return next state and rewards
                    full_state, full_reward, done, info = env.step(total_action)

                    true_next_state = full_state[0]
                    next_state = full_state[1]
                    param = full_state[2] #mass
                    
                    reward = full_reward[0]
                    reward_swing = full_reward[1]
                    

                    #next states for in the networks
                    next_state_pend = np.array([np.cos(next_state[0]), np.sin(next_state[0]), next_state[1]], dtype=np.float64)
                    action_swing_n = agent_pendulum.policy(next_state_pend, ou_noise)

                    #update system noises
                    angle_noise = np.random.normal(0,  0.01**2)
                    vel_noise = np.random.normal(0,  0.01**2)
                    env.set_angle_noise(angle_noise)
                    env.set_vel_noise(vel_noise)

                    if observed == "angle":
                        y_th_n = get_next_meas_angle(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                        next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32) 
                        y_th_est_n = next_state[0]

                    elif observed == "velocity":
                        y_th_n = get_next_meas_vel(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                        next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32)
                        y_th_est_n = next_state[1] 

                    else:
                        newth_n, newthdot_n = get_next_meas(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                        y_th_n=[newth_n,newthdot_n]
                        y_th_est_n = next_state
                        next_state_for_est = np.array([next_state[0], next_state[1], y_th_n[0], y_th_n[1]], dtype=np.float32)
                    

                    # record state, action, next state, reward in the buffer and train
                    agent.buffer.record(state_for_est, action, reward, next_state_for_est, done, y_th, y_th_est)
                    a_loss, c_loss, kld_value = agent.train(observed, training_epochs=TRAIN_EPOCHS, tmax=1000) # 1000 for velocity observed 0.01 2000 foe 0.1

                    ep_reward += reward
                    ep_reward_pend += reward_swing

                    # update current state and measurements
                    state = next_state
                    true_state = true_next_state

                    state_for_est = next_state_for_est
                    state_pend = next_state_pend
                    y_th = y_th_n
                    y_th_est = y_th_est_n
                    action_swing = action_swing_n

                    t += 1

                    if done:
                        ep_reward_list.append(ep_reward)
                        total_steps += t
                        ep_res_swing.append(ep_reward_pend)

                        #clear buffer pendulum if needed
                        # agent_pendulum.buffer.clear_buffer()

                        break

                else:
                   

                    # get action based on estimation
                    action = agent.policy(state_for_est, y_th, y_th_est, observed, greedy=True)

                    # total action
                    total_action = np.array([action[0][0], action[0][1], action_swing, y_th, action[0][2]])
                    full_state, full_reward, done, info = env.step(total_action)

                    true_next_state = full_state[0]
                    next_state = full_state[1]
                    param=full_state[2]
                    
                    reward = full_reward[0]
                    reward_swing = full_reward[1]

                    #next states for in the networks
                    next_state_pend = np.array([np.cos(next_state[0]), np.sin(next_state[0]), next_state[1]], dtype=np.float64)

                    action_swing_n = agent_pendulum.policy(next_state_pend, ou_noise)

                    angle_noise = np.random.normal(0,  0.01**2)
                    vel_noise = np.random.normal(0,  0.01**2)
                    env.set_angle_noise(angle_noise)
                    env.set_vel_noise(vel_noise)

                    if observed == "angle":
                        y_th_n = get_next_meas_angle(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                        next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32) 
                        y_th_est_n = next_state[0]

                    elif observed == "velocity":
                        y_th_n = get_next_meas_vel(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                        next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32)
                        y_th_est_n = next_state[1] 

                    else:
                        newth_n, newthdot_n = get_next_meas(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                        y_th_n=[newth_n,newthdot_n]
                        y_th_est_n = next_state
                        next_state_for_est = np.array([next_state[0], next_state[1], y_th_n[0], y_th_n[1]], dtype=np.float32)
                    

                    #learning pendulum swingup
                    agent_pendulum.buffer.record_ddpg((state_pend, action_swing, reward_swing, next_state_pend))
                    agent_pendulum.experience_replay()
                    agent_pendulum.update_target_models()



                    ep_reward += reward
                    ep_reward_pend += reward_swing

                    state = next_state
                    true_state = true_next_state

                    state_for_est = next_state_for_est
                    state_pend = next_state_pend

                    y_th = y_th_n
                    y_th_est = y_th_est_n
                    action_swing = action_swing_n

                    t += 1

                    if done:
                        ep_reward_list.append(ep_reward)
                        ep_res_swing.append(ep_reward_pend)
                        total_steps += t
                        
                        break

        print(ep, " Episode reward: ", round(ep_reward,1))
        print(ep, " Episode reward pendulum: ", round(ep_reward_pend,1))

        #validation
        test_score, test_score_pend, avg_len = validate(env, agent, agent_pendulum, observed)

        print("validation score pend: ", round(test_score_pend,1), " validation score: est: ", round(test_score,1))

        #break prematurely
        if ep>30 and end_training==False:
            break

        if end_training==False and best_score< test_score:
            #save best score
            best_score = test_score
            #save weights and rewards
            diff_x1_l = np.asarray(ep_res_swing)
            diff_x2_l = np.asarray(ep_reward_list)
            title1 = "./npzs/swing_rewardszz" + str(nr) 
            title2 = "./npzs/est_rewardszz" + str(nr)
            np.savez_compressed(title1, a=diff_x1_l)
            np.savez_compressed(title2, a=diff_x2_l)
            agent.save_model(path, 'actor_weightszz' + str(nr)+ '.h5', 'critic_weightszz' + str(nr) +'.h5')
            agent_pendulum.save_weights('actor_weights_pendzz' + str(nr) +'.h5', 'critic_weights_pendzz' + str(nr) +'.h5')
            print("saved models")

            diff_len = np.abs(1-avg_len)
            if best_score > thres and diff_len <0.03:
                end_training = True


        if best_score_pend < test_score_pend and end_training==True:
            print("Best score, update saved models")
            #save best score
            best_score_pend = test_score_pend
            #save weights and rewards
            agent.save_model(path, 'actor_weightszz' + str(nr)+ '.h5', 'critic_weightszz' + str(nr) +'.h5')
            agent_pendulum.save_weights('actor_weights_pendzz' + str(nr) +'.h5', 'critic_weights_pendzz' + str(nr) +'.h5')
            diff_x1_l = np.asarray(ep_res_swing)
            diff_x2_l = np.asarray(ep_reward_list)
            title1 = "./npzs/swing_rewardszz" + str(nr) 
            title2 = "./npzs/est_rewardszz" + str(nr)
            np.savez_compressed(title1, a=diff_x1_l)
            np.savez_compressed(title2, a=diff_x2_l)

            print('*** Episode: {}, validation_score: {}. Model saved. ***'.format(ep, best_score))


    #save rewards after training is finished
    diff_x1_l = np.asarray(ep_res_swing)
    diff_x2_l = np.asarray(ep_reward_list)
    title1 = "./npzs/swing_rewardszz" + str(nr) 
    title2 = "./npzs/est_rewardszz" + str(nr) 
    np.savez_compressed(title1, a=diff_x1_l)
    np.savez_compressed(title2, a=diff_x2_l)

    env.close()


# validation loop
def validate(env, agent, agent_pendulum, observed, ep_max=5):
    ep_reward_list = []
    ep_reward_list_pend = []
    action_param_list=[]

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))


    for ep in range(ep_max):
        #reset env
        true_state = env.reset()
        state = env.reset_est()
        state_pend = np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

        #get swing action
        state_pend = tf.squeeze(tf.expand_dims(tf.convert_to_tensor(state_pend), 0))
        action_swing = agent_pendulum.policy(state_pend, ou_noise, greedy=True)

        #system noise
        angle_noise = np.random.normal(0,  0.01**2)
        vel_noise = np.random.normal(0,  0.01**2)
        env.set_angle_noise(angle_noise)
        env.set_vel_noise(vel_noise)

        #init measurement and get input state est.
        if observed == "angle":
            y_th = get_next_meas_angle(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R)
            state_for_est = np.array([state[0], state[1], y_th], dtype=np.float32) 
            y_th_est = state[0]

        elif observed == "velocity":
            y_th = get_next_meas_vel(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R)
            state_for_est = np.array([state[0], state[1], y_th], dtype=np.float32)
            y_th_est = state[1] 

        else:
            newth, newthdot = get_next_meas(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R)
            y_th=[newth,newthdot]
            y_th_est = state
            state_for_est = np.array([state[0], state[1], y_th[0], y_th[1]], dtype=np.float32)

        ep_reward = 0
        ep_reward_pend = 0

        t = 0
        while True:
            env.render()

            #get greedy action state est.
            action = agent.policy(state_for_est, y_th, y_th_est, observed, greedy=True)

            #total action input for gym
            total_action = np.array([action[0][0], action[0][1], action_swing, y_th, action[0][2]])

            full_state, full_reward, done, info = env.step(total_action)

            true_next_state = full_state[0]
            next_state = full_state[1]
            param = full_state[2]
            action_param_list.append(param)
                    
            reward = full_reward[0]
            reward_swing = full_reward[1]
            
            #get next state input for pend
            next_state_pend = np.array([np.cos(next_state[0]), np.sin(next_state[0]), next_state[1]])

            # get new action from next est. state 
            action_swing_n = agent_pendulum.policy(next_state_pend, ou_noise, greedy=True)

            #update system noise
            angle_noise = np.random.normal(0,  0.01**2)
            vel_noise = np.random.normal(0,  0.01**2)
            env.set_angle_noise(angle_noise)
            env.set_vel_noise(vel_noise)

            if observed == "angle":
                y_th_n = get_next_meas_angle(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32) 
                next_state_for_param = next_state_for_est #np.array([next_state[0], next_state[1]],dtype=np.float32)
                y_th_est_n = next_state[0]

            elif observed == "velocity":
                y_th_n = get_next_meas_vel(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32)
                y_th_est_n = next_state[1] 

            else:
                newth_n, newthdot_n = get_next_meas(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                y_th_n=[newth_n,newthdot_n]
                y_th_est_n = next_state
                next_state_for_est = np.array([next_state[0], next_state[1], y_th_n[0], y_th_n[1]], dtype=np.float32)

            
            ep_reward += reward
            ep_reward_pend += reward_swing

            t += 1

            state = next_state
            true_state = true_next_state
            state_pend = next_state_pend
            state_for_est = next_state_for_est

            y_th = y_th_n 
            y_th_est = y_th_est_n
            action_swing = action_swing_n

            if done:
                ep_reward_list.append(ep_reward)
                ep_reward_list_pend.append(ep_reward_pend)


                break
    print(np.mean(action_param_list), np.std(action_param_list))
    return np.mean(ep_reward_list), np.mean(ep_reward_list_pend), np.mean(action_param_list) 



# test loop
def test(env, agent, agent_pendulum, observed, nr, R):
    list_actions=[]
    print(nr, R)

    if observed == "angle":
        env.set_angle("angle")
    if observed == "velocity":
        env.set_angle("velocity")
    if observed == "both":
        env.set_angle("both")

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    #load a model
    path = "./"
    act_w = 'actor_weightszz' +str(nr) + '.h5'
    cr_w = 'critic_weightszz' +str(nr) + '.h5'
    act_w_pend = 'actor_weights_pendzz'+str(nr)+'.h5'
    cr_w_pend = 'critic_weights_pendzz'+str(nr)+'.h5'
    agent.load_model(path, act_w, cr_w)
    agent_pendulum.load_weights(act_w_pend, cr_w_pend)

    ep_reward_list = []
    ep_reward_pend_list = []
    ep_reward_par_list = []

    #iterate over 50 different episodes
    for ep in range(50):

        list_actions_ep = []
        env.seed(ep)

        true_state = env.reset()
        state = env.reset_est()
        state_pend = np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

        #get swing action
        state_pend = tf.squeeze(tf.expand_dims(tf.convert_to_tensor(state_pend), 0))
        action_swing =agent_pendulum.policy(state_pend, ou_noise, greedy=True)

        #system noise
        angle_noise = np.random.normal(0,  0.01**2)
        vel_noise = np.random.normal(0,  0.01**2)
        env.set_angle_noise(angle_noise)
        env.set_vel_noise(vel_noise)

        if observed == "angle":
            y_th = get_next_meas_angle(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R)
            state_for_est = np.array([state[0], state[1], y_th], dtype=np.float32) 
            y_th_est = state[0]
            y_th_meas = -y_th+np.pi #for plotting purposes (different coord system gym)

        elif observed == "velocity":
            y_th = get_next_meas_vel(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R)
            state_for_est = np.array([state[0], state[1], y_th], dtype=np.float32)
            y_th_est = state[1] 
            y_th_meas = y_th

        else:
            newth, newthdot = get_next_meas(true_state[0], true_state[1], action_swing, angle_noise, vel_noise, R)
            y_th=[newth,newthdot]
            y_th_est = state
            state_for_est = np.array([state[0], state[1], y_th[0], y_th[1]], dtype=np.float32)
            y_th_meas = y_th

        theta_list = []
        thetadot_list = []
        time_list = []
        theta_est_list = []
        thetadot_est_list = []
        ydot_list = []
        y_list = []
        diff_x1_l =[]
        diff_x2_l =[]

        ep_reward = 0
        ep_reward_pend = 0
        ep_reward_par = 0
        t = 0
        time=0

        #FOR PLOTTING  
        theta = -true_state[0]+np.pi
        thetadot = true_state[1]
        theta_est = -state[0]+np.pi
        thetadot_est = state[1]

        theta_list.append(theta)
        thetadot_list.append(thetadot)
        theta_est_list.append(theta_est)
        thetadot_est_list.append(thetadot_est)
        time_list.append(time)
        y_list.append(y_th_meas)

        print(ep, " episode")
        while True:
            env.render()

            action = agent.policy(state_for_est, y_th, y_th_est, observed, greedy=True)

            total_action = np.array([action[0][0], action[0][1], action_swing, y_th, action[0][2]])

            full_state, full_reward, done, info = env.step(total_action)

            true_next_state = full_state[0]
            next_state = full_state[1]
            param = full_state[2]
                    
            reward = full_reward[0]
            reward_swing = full_reward[1]
            
            list_actions.append(param)
            list_actions_ep.append(param)

            next_state_pend = np.array([np.cos(next_state[0]), np.sin(next_state[0]), next_state[1]])

            action_swing_n =agent_pendulum.policy(next_state_pend, ou_noise, greedy=True)

            angle_noise = np.random.normal(0,  0.01**2)
            vel_noise = np.random.normal(0,  0.01**2)
            env.set_angle_noise(angle_noise)
            env.set_vel_noise(vel_noise)

            if observed == "angle":
                y_th_n = get_next_meas_angle(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32) 
                y_th_est_n = next_state[0]

            elif observed == "velocity":
                y_th_n = get_next_meas_vel(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                next_state_for_est = np.array([next_state[0], next_state[1], y_th_n], dtype=np.float32)
                y_th_est_n = next_state[1] 

            else:
                newth_n, newthdot_n = get_next_meas(true_next_state[0], true_next_state[1], action_swing_n, angle_noise, vel_noise, R)
                y_th_n=[newth_n,newthdot_n]
                y_th_est_n = next_state
                next_state_for_est = np.array([next_state[0], next_state[1], y_th_n[0], y_th_n[1]], dtype=np.float32)

            ep_reward += reward
            ep_reward_pend += reward_swing
            t += 1

            tf_th = -true_next_state[0] +np.pi
            tf_est_th = -next_state[0] +np.pi
            tf_thdot = true_next_state[1]
            tf_est_thdot = next_state[1]
            if observed == "angle":
                y_th_meas = -y_th +np.pi

            else:
                y_th_meas = y_th
            time+=env.dt

            diff_x1 =tf_th-tf_est_th
            diff_x2 = tf_thdot -tf_est_thdot
            diff_x1_l.append(diff_x1)
            diff_x2_l.append(diff_x2)

            time_list.append(time)
            theta_list.append(tf_th)
            thetadot_list.append(tf_thdot)
            theta_est_list.append(tf_est_th)
            thetadot_est_list.append(tf_est_thdot)
            y_list.append(y_th_meas)

            state = next_state
            true_state = true_next_state
            state_pend = next_state_pend
            state_for_est = next_state_for_est

            y_th = y_th_n 
            y_th_est = y_th_est_n
            action_swing = action_swing_n


            if done:
                ep_reward_list.append(ep_reward)
                ep_reward_pend_list.append(ep_reward_pend)

                print('Episode: {}, Reward: {}'.format(ep, ep_reward), ep_reward_pend)
                diff_x1_l = np.asarray(diff_x1_l)
                diff_x2_l = np.asarray(diff_x2_l)

                est_angle = np.asarray(theta_est_list)
                est_vel = np.asarray(thetadot_est_list)
                theta_list_np = np.asarray(theta_list)
                thetadot_list_np = np.asarray(thetadot_list)

                #saving
                title1 = "./npzs/wangle_x1_" +str(nr) + str(ep)
                title2 = "./npzs/wangle_x2_" +str(nr)+ str(ep)
                title5 = "./npzs/wangle_x1_est_" +str(nr)+ str(ep)
                title6 = "./npzs/wangle_x2_est" +str(nr)+ str(ep)
                title3 = "./npzs/wreal_angle" +str(nr)
                title4 = "./npzs/wreal_vel" +str(nr)
                np.savez_compressed(title1, a=diff_x1_l)
                np.savez_compressed(title2, a=diff_x2_l)
                np.savez_compressed(title3, a=theta_list_np)
                np.savez_compressed(title4, a=thetadot_list_np)
                np.savez_compressed(title5, a=est_angle)
                np.savez_compressed(title6, a=est_vel)

                print(np.mean(list_actions_ep), np.std(list_actions_ep))

                break
        #saving plots
        plt.figure(ep*nr)
        plt.plot(time_list, theta_list, color="red", label="Angle")
        plt.plot(time_list, thetadot_list, color="blue", label = "Angular velocity")
        plt.plot(time_list, theta_est_list, color="black", linestyle='dashed', label="Est. angle")
        plt.plot(time_list, thetadot_est_list, color="black",linestyle='dotted',label = "Est. angular velocity ")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad) and angular velocity (rad/s)")
        plt.legend()
        plt.savefig('./plots' +str(nr) + '/plot_test_pend' + str(ep) + '.png')


        plt.figure(ep*10000*nr)
        plt.plot(time_list, y_list, color="black", label = "Measurements ")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad) and angular velocity (rad/s)")
        plt.legend()
        plt.savefig('./plots' +str(nr) + '/measurement' + str(ep) + '.png')

    #print stats
    print('Avg episodic reward: ', np.mean(ep_reward_list),  np.std(ep_reward_list), np.mean(ep_reward_pend_list),  np.std(ep_reward_pend_list),np.mean(ep_reward_par_list), np.std(ep_reward_par_list) )
    # print(np.mean(list_actions), np.std(list_actions))
    env.close()





