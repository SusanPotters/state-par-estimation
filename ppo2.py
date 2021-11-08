"""
PPO Algorithm for Pendulum Gym Environment
Tensorflow 2.x compatible
- It seems to work properly with best episodic score reaching -200 within 1000 episodes or around 10-12 seasons
- Implements both 'KL-Penalty' method as well as 'PPO-Clip' method
- makes use of tensorflow probability
- The program terminates when the season score over 50 episodes > -200
"""
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
import gym_pendulum
############################

print('TFP Version:', tfp.__version__)
print('Tensorflow version:', tf.__version__)
print('Keras Version:', tf.keras.__version__)

# set random seed for reproducibility
tf.random.set_seed(20)
np.random.seed(20)


############### hyper parameters

TRAIN_EPOCHS = 60      # training epochs in each season
R = 10


#####################
# ACTOR NETWORK
####################
class Actor:
    def __init__(self, state_size, action_size,
                 learning_rate, epsilon, lmbda, kl_target,
                 upper_bound, method='clip'):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.upper_bound = upper_bound
        self.epsilon = epsilon  # required for 'clip' method
        self.lam = lmbda  # required for 'penalty' method
        self.method = method
        self.kl_target = kl_target  # required for 'penalty' method
        self.kl_value = 0       # most recent kld value

        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additions
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def _build_net(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        state_input = layers.Input(shape=self.state_size)
        l1 = layers.Dense(16, activation='relu', name="state")(state_input)

        # l2 = layers.Dense(8, activation='relu', name="par1")(state_input)
        # l3 = layers.Dense(8, activation='relu', name="par2")(l2)
        # l4 = layers.Dense(8, activation='relu', name="par3")(l3)

        # concat = layers.Concatenate(name="concat")([l1, l4])

        # out = layers.Dense(16, activation='relu', name="both")(concat)
        # l = layers.Dense(64, activation='relu')(l)
        # print(self.action_size[0]+1)
        # net_out1 = layers.Dense(self.action_size[0]-1, activation='relu',
        #                        kernel_initializer=last_init)(l1)

        net_out = layers.Dense(self.action_size[0], activation='relu',
                               kernel_initializer=last_init)(l1)

        # net_out = layers.Concatenate(name="concat")([net_out1, net_out2])

        net_out = net_out * self.upper_bound
        print(net_out)
        print(self.upper_bound)
        model = keras.Model(state_input, net_out)
        model.summary()
        return model

    def __call__(self, state):
        # input state is a tensor
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std  # returns tensors

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def train(self, state_batch, action_batch, advantages, old_pi, y_batch, y_est_batch, observed):

        with tf.GradientTape() as tape:
            action_batch = tf.squeeze(action_batch)
            m = tf.squeeze(self.model(state_batch))
        
            
            if observed == "angle" or observed=="velocity":

                y = tf.math.subtract(y_batch, y_est_batch)
                y = tf.expand_dims(y, 0)
                y1 = tf.ones_like(y)
                y_c= tf.concat([y, y],0)
                y_out = tf.transpose(tf.concat([y_c, y1], 0))
            else:
                y1 = tf.math.subtract(y_batch[:,0], y_est_batch[:,0])
                y2 = tf.math.subtract(y_batch[:,1], y_est_batch[:,1])
                y1 = tf.expand_dims(y1, 0)
                y2 = tf.expand_dims(y2, 0)
                y_out = tf.transpose(tf.concat([y1, y2], 0))

            mean = tf.math.multiply(m, y_out)

            std = tf.squeeze(tf.exp(self.model.logstd))
            # print(std, "std")
            pi = tfp.distributions.Normal(mean, std)

            ratio = tf.exp(pi.log_prob(tf.squeeze(action_batch)) -
                           old_pi.log_prob(tf.squeeze(action_batch)))


            # print(ratio.shape, advantages.shape)
            surr = ratio * advantages  # surrogate function
            kl = tfp.distributions.kl_divergence(old_pi, pi)
            self.kl_value = tf.reduce_mean(kl)
            if self.method == 'penalty':  # ppo-penalty method
                actor_loss = -(tf.reduce_mean(surr - self.lam * kl))
                # # update the lambda value after each epoch
                # if kl_mean < self.kl_target / 1.5:
                #   self.lam /= 2
                # elif kl_mean > self.kl_target * 1.5:
                #   self.lam *= 2
            elif self.method == 'clip':  # ppo-clip method
                actor_loss = - tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio,
                                                      1. - self.epsilon, 1. + self.epsilon) * advantages))
            actor_weights = self.model.trainable_variables

        # outside gradient tape
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))

        return actor_loss.numpy(), self.kl_value.numpy()

    def update_lambda(self):
        # update the lambda value after each epoch
        if self.kl_value < self.kl_target / 1.5:
            self.lam /= 2
        elif self.kl_value > self.kl_target * 1.5:
            self.lam *= 2


####################################
# CRITIC NETWORK
################################
class Critic:
    def __init__(self, state_size, action_size,
                 learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.train_step_count = 0
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model = self._build_net()

    def _build_net(self):
        state_input = layers.Input(shape=self.state_size)
        out = layers.Dense(16, activation="relu")(state_input)
        # out = layers.Dense(64, activation="relu")(out)
        # out = layers.Dense(64, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
        model.summary()
        return model

    def train(self, state_batch, disc_rewards):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = tf.squeeze(self.model(state_batch))
            critic_loss = tf.math.reduce_mean(tf.square(disc_rewards - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


######################
# BUFFER
######################
class Buffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_capacity)

    def __len__(self):
        return len(self.buffer)

    def record(self, state, action, reward, next_state, done, y_th, y_th_est):
        self.buffer.append([state, action, reward, next_state, done, y_th, y_th_est])

    def sample(self):
        valid_batch_size = min(len(self.buffer), self.batch_size)
        mini_batch = random.sample(self.buffer, valid_batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        y_th_batch =[]
        y_th_est_batch = []

        for i in range(valid_batch_size):
            state_batch.append(mini_batch[i][0])
            action_batch.append(mini_batch[i][1])
            reward_batch.append(mini_batch[i][2])
            next_state_batch.append(mini_batch[i][3])
            done_batch.append(mini_batch[i][4])
            y_th_batch.append(mini_batch[i][5])
            y_th_est_batch.append(mini_batch[i][6])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, y_th_batch, y_th_est_batch

    def save_data(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_data(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)

    def get_samples(self, n_samples=None):

        if n_samples is None or n_samples > len(self.buffer):
            n_samples = len(self.buffer)

        s_batch = []
        a_batch = []
        r_batch = []
        ns_batch = []
        d_batch = []
        y_batch =[]
        y_est_batch = []

        for i in range(n_samples):
            s_batch.append(self.buffer[i][0])
            a_batch.append(self.buffer[i][1])
            r_batch.append(self.buffer[i][2])
            ns_batch.append(self.buffer[i][3])
            d_batch.append(self.buffer[i][4])
            y_batch.append(self.buffer[i][5])
            y_est_batch.append(self.buffer[i][6])

        return s_batch, a_batch, r_batch, ns_batch, d_batch, y_batch, y_est_batch

    def clear(self):
        # empty the buffer
        self.buffer.clear()


#########################################
## PPO AGENT
########################################
class PPOAgent:
    def __init__(self, state_size, action_size, batch_size,
                 memory_capacity, upper_bound,
                 lr_a=1e-3, lr_c=1e-3,
                 gamma=0.99, lmbda=0.5, epsilon=0.2, kl_target=0.01,
                 method='clip'):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma  # discount factor
        self.upper_bound = upper_bound
        self.lmbda = lmbda  # required for GAE
        self.epsilon = epsilon  # required for PPO-CLIP
        self.kl_target = kl_target
        self.method = method
        self.best_ep_reward = -np.inf

        self.actor = Actor(self.state_size, self.action_size,
                           self.actor_lr, self.epsilon, self.lmbda,
                           self.kl_target, self.upper_bound, self.method)
        self.critic = Critic(self.state_size, self.action_size, self.critic_lr)
        self.buffer = Buffer(self.memory_capacity, self.batch_size)

    def policy(self, state, y_th, y_th_est, observed, greedy=False):
        tf_state = tf.expand_dims(tf.squeeze(tf.convert_to_tensor(state)), 0)
        out, std = self.actor(tf_state)
        mean = np.zeros_like(out)

        # print(mean)

        if observed == "angle" or observed == "velocity":
            
            mean[0] = out[0] * (y_th- y_th_est)
            mean[1] = out[1] * (y_th- y_th_est)
            mean[2] = out[2] 
        else:
            
            mean[0] = out[0] * (y_th[0]- y_th_est[0]) 
            mean[1] = out[1] * (y_th[1]- y_th_est[1])
            mean[2] = out[2] 

        if greedy:
            action = tf.squeeze(mean)
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = tf.squeeze(pi.sample(sample_shape=(1,)))
        action = tf.expand_dims(tf.convert_to_tensor(action),0)

        valid_action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)


        return valid_action.numpy()

    def train(self, observed, training_epochs=20, tmax=None):
        if tmax is not None and len(self.buffer) < tmax:
            # print(tmax, len(self.buffer))
            return 0, 0, 0
        n_split = len(self.buffer) // self.batch_size
        n_samples = n_split * self.batch_size

        print("training")

        s_batch, a_batch, r_batch, ns_batch, d_batch, y_batch, y_est_batch  = \
            self.buffer.get_samples(n_samples)

        r_batch = np.float32(r_batch)
        

        s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
        a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
        ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
        d_batch = tf.convert_to_tensor(d_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
        y_est_batch = tf.convert_to_tensor(y_est_batch, dtype=tf.float32)

        disc_sum_reward = PPOAgent.discount(r_batch.numpy(), self.gamma)
        advantages = self.compute_advantages(r_batch, s_batch,
                                             ns_batch, d_batch)  # returns a numpy array
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        disc_sum_reward = tf.convert_to_tensor(disc_sum_reward, dtype=tf.float32)

        # current policy
        mean, std = self.actor(s_batch)

        out = np.zeros_like(mean)
        if observed == "angle" or observed == "velocity":
            out[:,0] = mean[:,0] * (y_batch- y_est_batch)
            out[:,1] = mean[:,1] * (y_batch- y_est_batch) 
            out[:,2] = mean[:,2] 
        else:
            out[:,0] = mean[:,0] * (y_batch[:,0]- y_est_batch[:,0])
            out[:,1] = mean[:,1] * (y_batch[:,1]- y_est_batch[:,1])
            out[:,2] = mean[:,2] 

        mean=out

        pi = tfp.distributions.Normal(mean, std)

        s_split = tf.split(s_batch, n_split)
        a_split = tf.split(a_batch, n_split)
        dr_split = tf.split(disc_sum_reward, n_split)
        adv_split = tf.split(advantages, n_split)
        indexes = np.arange(n_split, dtype=int)
        y_split = tf.split(y_batch, n_split)
        y_est_split = tf.split(y_est_batch, n_split)

        a_loss_list = []
        c_loss_list = []
        kld_list = []
        np.random.shuffle(indexes)
        for _ in range(training_epochs):
            for i in indexes:
                old_pi = pi[i*self.batch_size: (i+1)*self.batch_size]

                # update actor
                a_loss, kld = self.actor.train(s_split[i], a_split[i], adv_split[i], old_pi, y_split[i], y_est_split[i], observed)
                a_loss_list.append(a_loss)
                kld_list.append(kld)

                # update critic
                c_loss_list.append(self.critic.train(s_split[i], dr_split[i]))

            # update lambda after each epoch
            if self.method == 'penalty':
                self.actor.update_lambda()

        actor_loss = np.mean(a_loss_list)
        critic_loss = np.mean(c_loss_list)
        mean_kld = np.mean(kld_list)

        # clear the buffer  -- this is important
        self.buffer.clear()

        return actor_loss, critic_loss, mean_kld

    @staticmethod
    def discount(x, gamma):
        return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def compute_advantages(self, r_batch, s_batch, ns_batch, d_batch):
        s_values = tf.squeeze(self.critic.model(s_batch))
        ns_values = tf.squeeze(self.critic.model(ns_batch))

        tds = r_batch + self.gamma * ns_values * (1. - d_batch) - s_values
        adv = PPOAgent.discount(tds.numpy(), self.gamma * self.lmbda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)   #sometimes helpful

        adv = np.reshape(adv, (adv.shape[0],1))
        adv_c = np.concatenate((adv, adv), axis=1)
        adv = np.concatenate((adv_c, adv), axis=1)

        return adv

    def save_model(self, path, actorfile, criticfile, bufferfile=None):
        actor_fname = path + actorfile
        critic_fname = path + criticfile

        self.actor.save_weights(actor_fname)
        self.critic.save_weights(critic_fname)

        if bufferfile is not None:
            buffer_fname = path + bufferfile
            self.buffer.save_data(buffer_fname)

    def load_model(self, path, actorfile, criticfile, bufferfile=None):

        actor_fname = path + actorfile
        critic_fname = path + criticfile

        self.actor.load_weights(actor_fname)
        self.critic.load_weights(critic_fname)

        if bufferfile is not None:
            buffer_fname = path + bufferfile
            self.buffer.load_data(buffer_fname)

        # print('Model Parameters are loaded ...')



