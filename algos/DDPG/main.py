# %%
import copy
import scipy
from utils.replay import ReplayBuffer
from core import DDPG
import torch
import time
import numpy as np
from fleet_env.env import FleetEnv
import argparse
from utils.logx import EpochLogger, setup_logger_kwargs
from utils.tool import test_agent
import os
script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_directory)
# %%
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--env_name', type=str, default='fleet')
parser.add_argument('--exp_name', type=str, default='fleet_DDPG')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--policy_noise_sigma', type=float, default=0.2)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--replay_size', type=int, default=1000000)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--start_steps', type=int, default=1000)
parser.add_argument('--save_epoch', type=int, default=10)
parser.add_argument('--steps_per_epoch', type=int, default=10000)
parser.add_argument('--num_steps', type=int, default=5000000)
# args = parser.parse_args(args=[])  # for jupyter
args = parser.parse_args()

leader_speed = scipy.io.loadmat('../../predict_meter.mat')['id_v_mat']
env = FleetEnv(sim_time_len=100,
               sim_time_step=0.01,
               leader_speed=leader_speed,
               follower_num=3,
               deceleration=0.001,
               distance_max_threshold=80,
               distance_min_threshold=1,
               velocity_threshold=20,
               init_distance=40)
env_test = copy.deepcopy(env)

env.seed(args.seed)
env_test.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=True, env_name=args.env_name)
logger = EpochLogger(**logger_kwargs)

args.obs_dim = env.observation_space.shape[0]
args.act_dim = env.action_space.shape[0]
logger.save_config(vars(args))

# %%
agent = DDPG(env.observation_space.shape[0],
             env.action_space,
             args.hidden_size,
             args.gamma,
             args.tau,
             args.lr,
             sigma=args.sigma
             )
replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=args.replay_size)

# %%
max_ep_len,  start_time = env._max_episode_steps,  time.time()
state,  done, ep_ret, ep_len = env.reset(),  False, 0, 0
for t in range(1, args.num_steps+1):
    if args.start_steps > t:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)

    next_state, reward, done, _ = env.step(action)
    ep_ret += reward
    ep_len += 1

    done = False if ep_len == max_ep_len else done

    replay_buffer.add(state, action, reward, next_state, mask=float(not done))
    state = next_state

    if done or (ep_len == max_ep_len):
        logger.store(EpRet=ep_ret, EpLen=ep_len)
        for j in range(ep_len):
            q_loss,  policy_loss, q_value = agent.update_parameters(replay_buffer, args.batch_size)
            logger.store(LossQ=q_loss,   LossPi=policy_loss, Qval=q_value)
        state,  done, ep_ret, ep_len = env.reset(),  False, 0, 0
        agent.expl_noise *= 0.999

    if t % args.steps_per_epoch == 0:
        epoch = t // args.steps_per_epoch
        test_agent(env_test, agent, logger, n=3)

        if epoch % args.save_epoch == 0:
            state_dic = {
                'critic': agent.critic.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
                'policy': agent.policy.state_dict(),
                'policy_target': agent.policy_target.state_dict()
            }
            logger.save_state(state_dic)

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', t)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('TestEpLen', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('Qval', average_only=True)

        remain_mins = ((args.num_steps-t)*(time.time()-start_time)/t)/60
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('remian_mins', remain_mins)

        logger.dump_tabular()
# %%
