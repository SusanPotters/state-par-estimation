'''
DDPG Algorithm for Pendulum-v0 environment
Source: https://keras.io/examples/rl/ddpg_pendulum/
Tensorflow 2.0 / Keras implementation
Gives a average reward of about -180 to -200 (over 40 episodes)
It has a single actor_critic class compared to the standard 3 classes, one for actor,
one for critic and one for the agent. This is the direct implementation available at the
above link.
'''
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import gym_pendulum
from ppo2 import * #PPOAgent, Buffer, Actor, Critic
print('Tensorflow version: ', tf.__version__)

# for reproducibility
random.seed(2212)
np.random.seed(2212)
tf.random.set_seed(2212)


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    print('Creating GIF Animation File. Wait ...')
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print('done!!')


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer_DDPG:
    def __init__(self, state_size, action_size, buffer_capacity=100000, batch_size=64):
        self.num_states = state_size
        self.num_actions = action_size
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record_ddpg(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def sample(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch

    def clear_buffer(self):
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))




class actor_critic_pend:
    def __init__(self, state_size, action_size,
                 critic_lr, actor_lr, gamma, tau,
                 upper_bound, lower_bound,
                 memory_capacity, batch_size):

        self.num_states = state_size
        self.num_actions = action_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        # training models
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        # target models
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Initially both models share same weights
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # create memory buffer
        self.buffer = Buffer_DDPG(self.num_states, self.num_actions,
                             self.memory_capacity, self.batch_size)

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh",
                               kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)

        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, curr_state, noise_object, greedy=False):
        curr_state = tf.expand_dims(curr_state, 0)
        sampled_actions = tf.squeeze(self.actor_model(curr_state))
        noise = noise_object()
        # Adding noise to action
        if greedy == True:
            sampled_actions = sampled_actions.numpy()
        else:
            sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return np.squeeze(legal_action)

    @staticmethod
    def _update_target(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def update_target_models(self):
        actor_critic_pend._update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        actor_critic_pend._update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def experience_replay(self):
        # sample a batch of experience from memory buffer
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()
        #print(state_batch, action_batch, reward_batch, next_state_batch)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch],
                                             training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables))



    def save_weights(self, actorfile, criticfile):
        self.actor_model.save_weights(actorfile)
        self.critic_model.save_weights(criticfile)


    def load_weights(self, actorfile, criticfile):
        self.actor_model.load_weights(actorfile)
        self.critic_model.load_weights(criticfile)

