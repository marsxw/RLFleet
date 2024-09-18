# %%
import fleet_env
import scipy
leader_speed = scipy.io.loadmat('predict_meter.mat')['id_v_mat']

env = fleet_env.FleetEnv(sim_time_len=100,
                         sim_time_step=0.01,
                         leader_speed=leader_speed,
                         follower_num=5,
                         deceleration=0.001,
                         distance_max_threshold=80,
                         distance_min_threshold=1,
                         velocity_threshold=20,
                         init_distance=40)
env.reset()
# %%
done = False
# for i in range(10000):
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if info['step'] % 100 == 0:
        env.render()  # 可视化结果

    if done:
        env.reset()
