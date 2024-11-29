'''
    This F1Wrapper is for 
        1. usage issue (gymnasium style)
        2. define observation & action space
        3. observation processing (concatenate lidar + other properties)
        
        ->  outputs of step function should have (n_agents, dim) shape
            ex) observations, rewards, terminates, truncates, infos = env.step([agent_1_action, 
                                                                                agent_2_action,
                                                                                ,,,
                                                                                agent_n_action])
'''

from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

EPS = 1e-8

def unnormalize_speed(value, minimum, maximum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = np.ones_like(value)*temp_a
    temp_b = np.ones_like(value)*temp_b
    return temp_a*value + temp_b


class F1Wrapper(gym.Wrapper):
    def __init__(self, args, maps, render_mode=None) -> None:
        
        self._env = gym.make("f1tenth_gym:f1tenth-v0", 
                            args=args,
                            maps=maps,
                            render_mode=render_mode)
        super().__init__(self._env)
        self.show_centerline = args.show_centerline

        # for control
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer
        self.noise_mean=0.0
        self.noise_std=0.01

        # for spaces
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.observation_space = spaces.Box(-np.inf*np.ones(self.obs_dim), np.inf*np.ones(self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        self.scan_history = deque(maxlen=3)

    def _reset_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]
        
        # cartesian coordinate pose
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = obs_dict['poses_theta'][0]
        
        # frenet coordinate pose
        poses_s, poses_d, poses_phi = 0, 0, 0
        s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)
        poses_s = s
        poses_d = ey
        poses_phi = phi
        self.position_frenet = np.stack([poses_s, poses_d]).T
        self.yaw_frenet = poses_phi

    def _step_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]
        
        # cartesian coordinate pose
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = obs_dict['poses_theta'][0]
        self.velocity = np.sqrt(obs_dict['linear_vels_x'][0]**2 + obs_dict['linear_vels_y'][0]**2)

        
        poses_s, poses_d, poses_phi = 0, 0, 0
        s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)
        poses_s = s
        poses_d = ey
        poses_phi = phi
        
        # assume that vehicle can't complete 1 cycle in single step
        self.delta_s = poses_s - self.position_frenet[0]
        total_track_s = self._env.track.centerline.spline.s[-1]
        if abs(self.delta_s) > total_track_s/2.0 and self.delta_s < 0:
            # if vehicle moved from spline endpoint to spline startpoint
            self.delta_s += total_track_s
                        
        # frenet coordinate pose
        self.position_frenet = np.stack([poses_s, poses_d]).T
        self.yaw_frenet = poses_phi

    def reset(self, **kwargs):
        self.history = deque(maxlen=10)
        obs_dict, info = self._env.reset(**kwargs)
        if self.show_centerline:
            self._env.unwrapped.add_render_callback(self._env.track.centerline.render_waypoints)
        self._reset_pose(obs_dict)
        obs = self.getObs(obs_dict, reset=True)
        self.scan_history = deque(maxlen=3)
        info['obs_dict'] = obs_dict
        return obs, info
    
    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own reward function.
    def calc_reward(self):        
        forward_reward = 0.0
        track_reward = 0.0
        collision_cost = 0.0

        # forward reward
        # forward_reward = self.delta_s
        # if self.velocity > 1.0:
        #     # forward_reward = min(self.velocity/2, 1.0)
        # else:
        #     forward_reward = -0.2
        if self.velocity > 0:
            forward_reward = math.exp(self.velocity) / math.exp(4)
        else:
            forward_reward = -0.2

   

        track_reward = abs(self.position_frenet[1])


        # danger_count = np.sum(self.scan < 0.6)
        # # 리워드 계산
        # gamma = 0.02  # 위험 영역 패널티 강도
        # track_reward = -gamma * danger_count

        # collison cost
        if self.collision:
            collision_cost = 5.0
        
        k1,k2,k3 = 1.0, -0.02, -1.0
        # final reward
        reward = k1 * forward_reward + k2 * track_reward + k3 * collision_cost 
        # print(forward_reward,track_reward, reward)

        # print(reward)
        reward_dict = {'forward_reward': forward_reward,
                       'collision_cost': collision_cost,
                       'track_reward': track_reward}
        return reward, reward_dict
    # ======================================================================== #
        
    def step(self, action:np.array):
        '''
            Original env's action : [desired steer, desired speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : ~inf ~ inf

            Changed to:             [desired steer, normalized speed]
                                    steer : -1(left) ~ 1(right)
                                    speed : -1 ~ 1              -> multiplied by self.max_speed
        '''
        _action = action.copy()
        _action[0] *= self.max_steer
        _action[1] = unnormalize_speed(_action[1], self.min_speed, self.max_speed)
        obs_dict, _, terminate, truncate, info = self._env.step(_action)
        self._step_pose(obs_dict)
        obs = self.getObs(obs_dict)
        reward, reward_dict = self.calc_reward()
        info['obs_dict'] = obs_dict
        info.update(reward_dict)
        return obs, reward, terminate, truncate, info
    
    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own observation.
    # def getObs(self, obs_dict, reset=False):
    #     '''
    #         Process LiDAR data and return observation with current and previous scans.
    #         LiDAR data is downsampled to 20 beams from the front 180-degree section.
    #     '''
    #     # LiDAR 스캔 데이터 가져오기
    #     scan = np.array(obs_dict['scans']).reshape(-1)  # raw lidar value (1080,)

    #     # 전방 180도 (270도 기준 중앙 180도 추출)
    #     start_index = 180  # 270도 시작
    #     end_index = 900    # 450도 끝
    #     front_scan = scan[start_index:end_index]

    #     # 20개 빔으로 축소 (등간격 샘플링)
    #     num_beams = 20
    #     beam_indices = np.linspace(0, len(front_scan) - 1, num=num_beams, dtype=int)
    #     reduced_scan = front_scan[beam_indices]

    #     # 가우시안 노이즈 추가
    #     noise = np.random.normal(self.noise_mean, self.noise_std, reduced_scan.shape)
    #     noisy_scan = reduced_scan + noise

    #     # 클리핑 및 정규화
    #     clipped_scan = np.clip(noisy_scan, 0, 10)  # 0에서 10 사이로 클리핑
    #     normalized_scan = clipped_scan / 10.0  # 0에서 1 사이로 정규화

    #     # 현재 LiDAR 데이터 저장
    #     self.scan = normalized_scan

    #     # 초기 상태 처리
    #     if len(self.scan_history) == 0:
    #         # 초기 상태: 히스토리를 현재 스캔 데이터로 채움
    #         self.scan_history.extend([normalized_scan] * 2)  # 총 2개로 히스토리 초기화
    #     else:
    #         # 기존 히스토리에 현재 데이터 추가
    #         self.scan_history.append(normalized_scan)

    #     # 이전 1개 LiDAR 데이터와 현재 데이터 병합 (총 40차원)
    #     previous_scans = np.concatenate(list(self.scan_history))

    #     # 최종 관측 값 반환
    #     observation = previous_scans
    #     return observation


    def getObs(self, obs_dict, reset=False):
        '''
            Original observation is dictionary with keys:
                === key ====     === shape ===
                'ego_idx'       | int
                'scans'         | (num_agents, 1080)
                'poses_x'       | (num_agents,)
                'poses_y'       | (num_agents,)
                'poses_theta'   | (num_agents,)
                'linear_vels_x' | (num_agents,)
                'linear_vels_y' | (num_agents,)
                'ang_vels_z'    | (num_agents,)
                'collisions'    | (num_agents,)
                'lap_times'     | (num_agents,)
                'lap_counts'    | (num_agents,)
            
            ==> Changed to include Gaussian noise, clipped, and normalized scans with previous scans
        '''
        # LiDAR 스캔 데이터 가져오기
        scan = np.array(obs_dict['scans']).reshape(-1)  # raw lidar value (1080,)

        # 가우시안 노이즈 추가
        noise = np.random.normal(self.noise_mean, self.noise_std, scan.shape)  # normal 또는 uniform으로 해보자
        noisy_scan = scan + noise  # 노이즈가 추가된 스캔 데이터

        # 클리핑 및 정규화
        clipped_scan = np.clip(noisy_scan, 0, 10)  # 0에서 10 사이로 클리핑
        normalized_scan = clipped_scan / 10.0  # 0에서 1 사이로 정규화

        # 현재 LiDAR 데이터 저장
        self.scan = normalized_scan

        # 초기 상태 처리
        if len(self.scan_history) == 0:
            # 초기 상태: 히스토리를 현재 스캔 데이터로 채움
            self.scan_history.extend([normalized_scan] * 3)
        else:
            # 기존 히스토리에 현재 데이터 추가
            self.scan_history.append(normalized_scan)

        # 이전 3개 LiDAR 데이터 병합
        previous_scans = np.concatenate(list(self.scan_history))

        # 최종 관측 값 반환
        observation = np.concatenate([normalized_scan, previous_scans])
        return observation
    # ======================================================================== #

    # def getObs(self, obs_dict, reset=False):
    #     '''
    #         Original observation is dictionary with keys:
    #             === key ====     === shape ===
    #             'ego_idx'       | int
    #             'scans'         | (num_agents, 1080)
    #             'poses_x'       | (num_agents,)
    #             'poses_y'       | (num_agents,)
    #             'poses_theta'   | (num_agents,)
    #             'linear_vels_x' | (num_agents,)
    #             'linear_vels_y' | (num_agents,)
    #             'ang_vels_z'    | (num_agents,)
    #             'collisions'    | (num_agents,)
    #             'lap_times'     | (num_agents,)
    #             'lap_counts'    | (num_agents,)
            
    #         ==> Changed to include Gaussian noise, clipped, and normalized scans
    #     '''
    #     # LiDAR 스캔 데이터 가져오기
    #     scan = np.array(obs_dict['scans']).reshape(-1)  # raw lidar value (1080,)

    #     # 가우시안 노이즈 추가
    #     noise = np.random.normal(self.noise_mean, self.noise_std, scan.shape)  #normal 또는 uniform으로 해보자
    #     noisy_scan = scan + noise  # 노이즈가 추가된 스캔 데이터

    #     # 클리핑 및 정규화
    #     clipped_scan = np.clip(noisy_scan, 0, 10)  # 0에서 10 사이로 클리핑
    #     normalized_scan = clipped_scan / 10.0  # 0에서 1 사이로 정규화
    #     self.scan=normalized_scan

    #     observation = normalized_scan
        
    #     return observation

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()