import numpy as np
from gym import spaces
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class FleetEnv:
    """
    自定义的车队环境
    """

    def __init__(self, sim_time_len=100,
                 sim_time_step=0.01,
                 leader_speed=None,
                 follower_num=3,
                 deceleration=0.001,
                 distance_max_threshold=80,
                 distance_min_threshold=1,
                 velocity_threshold=20,
                 init_distance=40,
                 seed=None
                 ):
        '''
            sim_time_len: 仿真时间长度
            time_step: 仿真步长
            leader_speed: 车队领航者的速度
            follower_num: 车队的跟随者数量
            deceleration: 每秒汽车速度衰减值
            distance_max_threshold: done时，最大的车间距
            distance_min_threshold: done时，最小的车间距
            velocity_threshold: done时，后车比前车快的最大的速度差
            init_distance: 初始化时每台车的距离
        '''
        self.sim_time_len, self.sim_time_step = sim_time_len, sim_time_step
        self.leader_speed = leader_speed  # 领航车的真实数据
        self.follower_num = follower_num
        self.distance_max_threshold = distance_max_threshold
        self.distance_min_threshold = distance_min_threshold
        self.velocity_threshold = velocity_threshold
        self.init_distance = init_distance
        self._max_episode_steps = sim_time_len*1/sim_time_step  # 最大步数

        self.A, self.B, self.C, self.D = self._generate_matrix(follower_num, deceleration, sim_time_step)

        self.action_space = spaces.Box(low=-4, high=4, shape=(follower_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1+follower_num+follower_num,), dtype=np.float32)

        self.leader_speed_sample = None  # 采样用于播放的领导车速度
        self.leader_position, self.leader_velocity = None, None
        self.state = np.zeros(2*follower_num)  # 状态空间
        self.sim_time = None  # 仿真时间
        self.step_num = None  # 环境步数
        self.action = np.zeros(follower_num)
        self.done = None

        # render使用
        self.fig, self.ax = None, None
        self.animation_started = False  # 用于标志动画是否已经启动
        self.colors = plt.cm.viridis(np.linspace(0, 1, self.follower_num + 1))  # 为每辆车分配不同的颜色

    def seed(self, seed=None):
        '''
            设置随机种子
        '''
        if seed:
            np.random.seed(seed)

    def _generate_matrix(self, action_dim, deceleration, sim_time_step):
        '''
            动态生成状态空间矩阵 A 和 B
            action_dim: 动作输入维度
        '''
        state_dim = 2*action_dim
        A = np.zeros((state_dim, state_dim))
        for i in range(0, state_dim, 2):
            A[i, i+1] = 1

        B = np.zeros((state_dim, action_dim))
        for i in range(action_dim):
            B[2*i+1, i] = 1

        C = np.eye(state_dim)
        for i in range(action_dim):
            C[2*i+1, 2*i+1] = 1  # - deceleration * sim_time_step

        D = np.zeros((state_dim, action_dim))
        return A, B, C, D

    def _scale_reward(self, reward, min_value, max_value):
        '''
            归一化reward
        '''
        reward = np.clip(reward, min_value, max_value)
        return (reward - min_value) / (max_value - min_value)

    def _getInterp(self, t, time_points, values):
        '''
            获取时间序列和值的插值
            t: 时间
            time_points: 时间序列
            values: 值序列
        '''
        return np.interp(t, time_points, values)

    def _get_positions_velocities_interval_v_delta(self):
        '''
            获取所有车的行驶距离、速度, 后车与前车的距离、速度差
        '''
        positions = np.concatenate((np.array([self.leader_position]), self.state[::2]))
        velocities = np.concatenate((np.array([self.leader_velocity]), self.state[1::2]))
        interval = np.array([positions[i] - positions[i+1] for i in range(self.follower_num)])  # 计算距离差
        v_delta = np.array([velocities[i+1] - velocities[i] for i in range(self.follower_num)])  # 计算速度差
        return positions, velocities, interval, v_delta

    def _get_observation(self):
        positions, velocities, interval, v_delta = self._get_positions_velocities_interval_v_delta()
        return np.concatenate((velocities,  interval))

    def _done(self):
        positions, velocities, interval, v_delta = self._get_positions_velocities_interval_v_delta()
        done_distance_max = (interval > self.distance_max_threshold).any()
        done_distance_min = (interval < self.distance_min_threshold).any()
        done_velocity_delta = (v_delta > self.velocity_threshold).any()
        # done_velocity = (velocities < 0).any()
        done_velocity = 0
        self.done = bool(done_distance_max or done_distance_min or done_velocity_delta or done_velocity)

        # if self.done:
        #     print(done_distance_max, done_distance_min, done_velocity_delta, done_velocity, self.step_num)
        return self.done

    def _reward(self):
        positions, velocities, interval, v_delta = self._get_positions_velocities_interval_v_delta()

        # 计算车距奖励
        delta_d = 10
        R_distance = 0
        for d, v in zip(interval, velocities[1:]):
            d_brake = 1.3 * v + 10
            if d < d_brake:
                R_distance -= (d_brake - d)
            elif d > d_brake + delta_d:
                R_distance -= (d - d_brake - delta_d)
        R_distance = self._scale_reward(R_distance, -15 * self.follower_num, 0)

        # 速度奖励
        R_speed = -np.abs(v_delta).sum()
        R_speed = self._scale_reward(R_speed, -10 * self.follower_num, 0)

        # 平滑奖励
        R_smoothness = -np.abs(self.action).sum()
        R_smoothness = self._scale_reward(R_smoothness, -self.action_space.shape[0] * 3, 0)

        reward = 0.5 * R_distance + 0.3 * R_speed + 0.2 * R_smoothness
        return reward

    def reset(self):
        """
        重置环境
        """
        start_index = np.random.randint(0, len(self.leader_speed) - self.sim_time_len)

        self.leader_speed_sample = copy.deepcopy(self.leader_speed[start_index:start_index+self.sim_time_len])
        self.leader_speed_sample[:, 0] -= self.leader_speed_sample[0, 0]  # 重置self.leader_speed_sample 时间轴

        self.leader_position = self.follower_num * self.init_distance + np.random.uniform(-10, 10)
        self.leader_velocity = self.leader_speed_sample[0, 1]

        # 初始化后车的位置 速度
        position = np.array([i*self.init_distance for i in range(self.follower_num)][::-1]) + np.random.uniform(-10, 10, self.follower_num)
        InitialCondition = np.array([])
        for i in range(self.follower_num):
            # 将位置和速度分别加入到 InitialCondition 中
            InitialCondition = np.append(InitialCondition, [position[i], self.leader_velocity])
        self.state = np.array(InitialCondition, dtype=np.float32)
        self.sim_time = 0
        self.step_num = 0
        self.action = np.zeros(self.follower_num, dtype=np.float32)
        self.done = False
        return self._get_observation()

    def step(self, action):
        """
        进行一步仿真
        """
        self.sim_time += self.sim_time_step
        self.step_num += 1

        # 更新领航者位置速度
        self.leader_velocity = self._getInterp(self.sim_time, self.leader_speed_sample[:, 0], self.leader_speed_sample[:, 1])
        self.leader_position += self.leader_velocity * self.sim_time_step

        # 更新后车位置速度
        self.action = np.clip(action, self.action_space.low, self.action_space.high)
        # 状态转移方程
        # x_prime = A * x + B * u * dt
        # y= C * x + D * u
        self.state = self.state + (self.A @ self.state + self.B @ self.action) * self.sim_time_step
        self.state = self.C @ self.state + self.D @ self.action  # 输入矩阵C里面有sim_time_step的速度衰减

        return self._get_observation(), self._reward(), self._done(), {'step': self.step_num}

    def render(self, refresh_interval=10):
        """
        动态渲染车队状态，显示每辆车的位置、编号、速度和与前车的距离。
        refresh_interval: 动画刷新时间间隔，单位毫秒
        """
        if not self.animation_started:
            positions, velocities, distances, _ = self._get_positions_velocities_interval_v_delta()
            # 初始化图形和动画（只运行一次）
            self.fig, self.ax = plt.subplots(figsize=(10, 2))
            self.ax.clear()
            self.ax.set_xlim(-10, max(positions) + 10)
            self.ax.set_ylim(-1, 1)

            self.scatters = self.ax.scatter(positions, [0] * len(positions), s=100, c=self.colors[:len(positions)])
            self.texts = [self.ax.text(positions[i], 0.1, "", fontsize=10, ha='center') for i in range(len(positions))]
            self.step_text = self.ax.text(0.05, 0.05, '', transform=self.ax.transAxes)

            self.ani = FuncAnimation(self.fig, self.update, frames=range(100), interval=refresh_interval, blit=False)
            plt.show(block=False)  # 非阻塞模式显示图像
            self.animation_started = True  # 标志动画已启动
        if self.step_num % 100 == 0:
            plt.pause(0.001)  # 短暂暂停以更新图像
        if self.done:
            for i in range(10):
                plt.pause(0.001)  # 更新最后图像

    def update(self, frame):
        # 获取更新后的状态
        positions, velocities, distances, _ = self._get_positions_velocities_interval_v_delta()

        # 更新散点的位置
        self.scatters.set_offsets(np.c_[positions, [0] * len(positions)])

        # 动态调整 x 轴范围
        self.ax.set_xlim(min(positions) - 10, max(positions) + 10)

        # 更新每辆车的编号、速度和与前车距离的文本信息
        for i in range(len(positions)):
            if i == 0:
                self.texts[i].set_position((positions[i], 0.1))
                self.texts[i].set_text(f"Car 1\nVel: {velocities[i]:.2f}")
            else:
                self.texts[i].set_position((positions[i], 0.1))
                self.texts[i].set_text(f"Car {i+1}\nVel: {velocities[i]:.2f}\nDist: {distances[i-1]:.2f}\nAction: {self.action[i-1]:.2f}")

        # 更新 step 数量
        self.step_text.set_text(f'Step: {self.step_num}')  # 更新左下角显示的 step 数量

        return self.scatters, self.texts, self.step_text
