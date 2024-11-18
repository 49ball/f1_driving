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

        # for spaces
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.observation_space = spaces.Box(-np.inf*np.ones(self.obs_dim), np.inf*np.ones(self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)

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

        # print(f"Cartesian Position (x, y): {self.position}")
        # print(f"Yaw (Cartesian): {self.yaw}")
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
        # print(f"Frenet Position (s, d): {self.position_frenet}")
        # print(f"Yaw (Frenet): {self.yaw_frenet}")
        # print(f"Delta s: {self.delta_s}")

    def reset(self, **kwargs):
        self.history = deque(maxlen=10)
        obs_dict, info = self._env.reset(**kwargs)
        if self.show_centerline:
            self._env.unwrapped.add_render_callback(self._env.track.centerline.render_waypoints)
        self._reset_pose(obs_dict)
        obs = self.getObs(obs_dict, reset=True)
        info['obs_dict'] = obs_dict
        return obs, info
    
    # ============================ implement here ============================ #
    # TODO
    # You need to implement your own reward function.
    # def calc_reward(self):        
    #     # forward reward
    #     forward_reward = 0.0
    #     track_reward = 0.0
    #     collision_cost = 0.0

    #     if self.delta_s > 0:
    #         forward_reward = 1.0
    #     else:
    #         forward_reward = -0.5
    #     # track reward
    #     if self.position_frenet[1] == 0:
    #         track_reward = 1.0

    #     # collison cost
    #     if self.collision:
    #         collision_cost = 20.0
        
    #     k1,k2,k3 = 1.0, 0.0, 1.0
    #     # final reward
    #     reward = k1 * forward_reward + k2 * track_reward - k3 * collision_cost
    #     print(reward)
    #     reward_dict = {'forward_reward': forward_reward,
    #                    'collision_cost': collision_cost,
    #                    'track_reward': track_reward}
        
    #     return reward, reward_dict
    def calc_reward(self):
        """
        Calculate reward based on the vehicle's current state.
        """
        # Initialize rewards and penalties
        forward_reward = 0.0
        track_reward = 0.0
        collision_cost = 0.0
        speed_reward = 0.0
        yaw_reward = 0.0

        # Parameters for scaling rewards
        k_forward = 1.0     # Weight for forward progress
        k_track = 1.0       # Weight for track alignment
        k_collision = -10.0 # Penalty for collision
        k_speed = 0.5       # Weight for maintaining speed
        k_yaw = 0.5         # Weight for yaw alignment

        # Forward progress reward (based on Frenet delta_s)
        if self.delta_s > 0:
            forward_reward = self.delta_s * k_forward
        else:
            forward_reward = -0.5  # Penalize if moving backward

        # Track alignment reward (based on Frenet position d)
        track_deviation = abs(self.position_frenet[1])  # Deviation from center (d)
        track_reward = max(0, 1 - track_deviation) * k_track  # Higher reward closer to center

        # Collision penalty
        if self.collision:
            collision_cost = k_collision

        # Speed reward (encourage maintaining a target speed, e.g., 10 m/s)
        target_speed = self.max_speed  # Target speed in m/s
        current_speed = np.linalg.norm(self.velocity)  # Vehicle speed
        speed_reward = max(0, 1 - abs(current_speed - target_speed) / target_speed) * k_speed

        # Yaw alignment reward (encourage alignment with road direction)
        yaw_deviation = abs(self.yaw_frenet)  # Deviation in Frenet yaw
        yaw_reward = max(0, 1 - yaw_deviation / np.pi) * k_yaw

        # Final reward
        reward = forward_reward + track_reward + speed_reward + yaw_reward + collision_cost

        # # Print debug information
        # print(f"Reward Breakdown:")
        # print(f"  Forward Reward: {forward_reward}")
        # print(f"  Track Reward: {track_reward}")
        # print(f"  Speed Reward: {speed_reward}")
        # print(f"  Yaw Reward: {yaw_reward}")
        # print(f"  Collision Cost: {collision_cost}")
        # print(f"  Total Reward: {reward}")

        # Create a detailed reward dictionary
        reward_dict = {
            'forward_reward': forward_reward,
            'track_reward': track_reward,
            'speed_reward': speed_reward,
            'yaw_reward': yaw_reward,
            'collision_cost': collision_cost,
            'total_reward': reward
        }

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
            
            ==> Changed to include only scans
        '''
        scan = np.array(obs_dict['scans']).reshape(-1)      # raw lidar value
        observation = scan

        return observation
    # ======================================================================== #

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()