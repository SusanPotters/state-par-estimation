from gym.envs.registration import register

register(
    id='MyPendulum-v2',
    entry_point='gym_pendulum.envs:PendulumEnvAdjusted',
    max_episode_steps=1000,
)
