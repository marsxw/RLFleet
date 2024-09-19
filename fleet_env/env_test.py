# %%
import time
import os
import env
import scipy
# %matplotlib qt

script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_directory)
leader_speed = scipy.io.loadmat('../predict_meter.mat')['id_v_mat']

env = env.FleetEnv(sim_time_len=100,
                   sim_time_step=0.01,
                   leader_speed=leader_speed,
                   follower_num=3,
                   deceleration=0.001,
                   distance_max_threshold=80,
                   distance_min_threshold=1,
                   velocity_threshold=20,
                   init_distance=40)
# %%
for _ in range(100):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(action, observation, reward, done, info)
        env.render()
    time.sleep(3)
