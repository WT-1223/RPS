import gym
from gym.envs.registration import register

register(
    id = 'lawyer_suspector_env_v0',
    entry_point='Lawyer-Suspector-Env.lawyer-suspector-env:LawyerSuspectEnv',
)

env = gym.spec('lawyer_suspector_env_v0')
print(env)