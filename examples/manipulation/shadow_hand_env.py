import torch
import math
import os
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

class ShadowHandEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg["num_privileged_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_dofs = env_cfg["num_dofs"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # reward parameters
        self.rot_eps = reward_cfg["rot_eps"]
        self.success_tolerance = reward_cfg["success_tolerance"]
        self.reach_goal_bonus = reward_cfg["reach_goal_bonus"]
        self.fall_dist = reward_cfg["fall_dist"]
        self.fall_penalty = reward_cfg["fall_penalty"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add hand
        self.hand_init_pos = torch.tensor(self.env_cfg["hand_init_pos"], device=self.device)
        self.hand_init_quat = torch.tensor(self.env_cfg["hand_init_quat"], device=self.device)
        self.inv_hand_init_quat = inv_quat(self.hand_init_quat)
        self.hand = self.scene.add_entity(
            gs.morphs.MJCF(
                file='xml/shadow_hand/left_hand.xml',
                #convexify=False,
            ),
        )
        self.object_init_pos = torch.tensor([0.05, -0.01, 0.41], device=self.device)
        self.target_init_pos = torch.tensor([0.0, 0.0, 0.6], device=self.device)
        random_object_init_quat = generate_random_quaternions(1, device=self.device)
        random_target_init_quat = generate_random_quaternions(1, device=self.device)
        self.object_init_quat = torch.tensor(random_object_init_quat, device=self.device)
        self.target_init_quat = torch.tensor(random_target_init_quat, device=self.device)

        self.object = self.scene.add_entity(
            gs.morphs.MJCF(
                pos = (0.1, 0.0, 0.2),
                scale = 0.0225,
                file="xml/Rubic_cube/Rubic_cube.xml",
            ),
        )

        self.target = self.scene.add_entity(
            gs.morphs.MJCF(
                pos = (0.0, 0.0, 0.4),
                scale = 0.0225,
                file="xml/Rubic_cube/Rubic_cube_fixed.xml",
            ),
        )

        # self.num_fingertips = 4
        # self.fingertip_pos = torch.zeros((self.num_envs, self.num_fingertips * 3), device=self.device, dtype=gs.tc_float)

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.all_dofs = [self.hand.get_joint(name).dof_idx_local for name in self.env_cfg["all_dof_names"]]
        self.motor_dofs = [self.hand.get_joint(name).dof_idx_local for name in self.env_cfg["action_dof_names"]]
        self.all_dof_pos_limits = torch.stack(self.hand.get_dofs_limit(self.all_dofs), dim=1)

        # PD control parameters
        self.hand.set_dofs_kp([self.env_cfg["kp"]] * self.num_dofs, self.motor_dofs)
        self.hand.set_dofs_kv([self.env_cfg["kd"]] * self.num_dofs, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.object_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.object_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        # actions
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        # hand
        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float)
        self.hand_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.hand_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        # object
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.object_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        #target
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        # distance
        self.goal_dist = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.quat_diff = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.rot_dist = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["all_dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.action_scale = self.env_cfg["action_scale"]
        self._prev_applied_actions = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device)
        self.extras = dict()  # extra information for logging

    def step(self, actions):
        # NaNチェックを追加
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        # control robot dofs
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        self.process_actions(clip_actions=self.actions)  # 修正: self.action -> self.actions
        self.hand.control_dofs_position(self._processed_actions, self.motor_dofs)

        self.scene.step()

        # NaNを検出して修正するためのヘルパー関数
        def safe_update(tensor, new_values):
            tensor[:] = new_values
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # update buffers with NaN防止
        self.episode_length_buf += 1
        self.hand_pos = safe_update(self.hand_pos, self.hand.get_pos())
        self.hand_quat = safe_update(self.hand_quat, self.hand.get_quat())
        self.dof_pos = safe_update(self.dof_pos, self.hand.get_dofs_position(self.all_dofs))
        self.dof_vel = safe_update(self.dof_vel, self.hand.get_dofs_velocity(self.all_dofs))

        # まず位置と四元数を更新 (NaNチェック付き)
        self.target_pos = safe_update(self.target_pos, self.target.get_pos())
        self.object_pos = safe_update(self.object_pos, self.object.get_pos())
        self.target_quat = safe_update(self.target_quat, self.target.get_quat())
        self.object_quat = safe_update(self.object_quat, self.object.get_quat())
        self.object_lin_vel = safe_update(self.object_lin_vel, self.object.get_vel())
        self.object_ang_vel = safe_update(self.object_ang_vel, self.object.get_ang())

        # 四元数の正規化
        self.hand_quat = self.hand_quat / (torch.norm(self.hand_quat, dim=1, keepdim=True) + 1e-6)
        self.object_quat = self.object_quat / (torch.norm(self.object_quat, dim=1, keepdim=True) + 1e-6)
        self.target_quat = self.target_quat / (torch.norm(self.target_quat, dim=1, keepdim=True) + 1e-6)

        # その後距離と差分を計算
        self.goal_dist = torch.norm(self.object_pos - self.target_pos, p=2, dim=-1)
        self.goal_dist = torch.nan_to_num(self.goal_dist, nan=0.0, posinf=10.0, neginf=0.0)

        self.quat_diff = quaternion_difference(self.object_quat, self.target_quat)
        self.rot_dist = rotation_distance(self.object_quat, self.target_quat)

        # リセットの条件確認
        reset_target_pose_mask = torch.isfinite(self.rot_dist) & (self.rot_dist <= self.success_tolerance)
        reset_target_pose_envs_ids = reset_target_pose_mask.nonzero(as_tuple=False).flatten()

        if len(reset_target_pose_envs_ids) > 0:
            self.reset_target_pose(reset_target_pose_envs_ids)

        # 落下検出
        out_of_reach_mask = torch.isfinite(self.goal_dist) & (self.goal_dist > self.fall_dist)
        out_of_reach_envs_ids = out_of_reach_mask.nonzero(as_tuple=False).flatten()

        if len(out_of_reach_envs_ids) > 0:
            self.reset_idx(out_of_reach_envs_ids)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward (NaN防止を各報酬関数に組み込み済み)
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # 最終的に報酬のNaNチェック
        self.rew_buf = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.dof_pos * self.obs_scales["dof_pos"], # 24
                self.object_pos * self.obs_scales["object_pos"], # 3
                self.object_quat * self.obs_scales["object_quat"], # 4
                self.target_pos * self.obs_scales["target_pos"], # 3
                self.target_quat * self.obs_scales["target_quat"], # 4
                self.quat_diff * self.obs_scales["quat_diff"], # 4
                self.last_actions * self.obs_scales["last_actions"], # 24
            ],
            axis=-1,
        )
        self.privileged_obs_buf = torch.cat(
            [
                self.dof_pos, # 24
                self.dof_vel, # 24
                self.object_pos, # 3
                self.object_quat, # 4
                self.object_lin_vel, # 3
                self.object_ang_vel, # 3
                self.target_pos, # 3
                self.target_quat, # 4
                self.quat_diff, # 4
                self.last_actions, # 24
            ],
            axis=-1,
        )

        # 観測のNaNを徹底的にチェック
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self.privileged_obs_buf = torch.nan_to_num(self.privileged_obs_buf, nan=0.0, posinf=0.0, neginf=0.0)

        # 数値範囲を制限
        self.obs_buf = torch.clamp(self.obs_buf, -10.0, 10.0)
        self.privileged_obs_buf = torch.clamp(self.privileged_obs_buf, -10.0, 10.0)

        # アクションの記録
        self.last_actions[:] = self.actions[:]

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0

        # 位置と速度のNaNチェック
        self.dof_pos = torch.nan_to_num(self.dof_pos, nan=0.0, posinf=0.0, neginf=0.0)
        self.dof_vel = torch.nan_to_num(self.dof_vel, nan=0.0, posinf=0.0, neginf=0.0)

        self.hand.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset hand
        self.hand_pos[envs_idx] = self.hand_init_pos
        self.hand_quat[envs_idx] = self.hand_init_quat.reshape(1, -1)

        # NaNチェック
        self.hand_pos = torch.nan_to_num(self.hand_pos, nan=0.0, posinf=0.0, neginf=0.0)
        self.hand_quat = torch.nan_to_num(self.hand_quat, nan=0.0, posinf=0.0, neginf=0.0)

        self.hand.set_pos(self.hand_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.hand.set_quat(self.hand_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.hand.zero_all_dofs_velocity(envs_idx)

        # reset object
        self.object_pos[envs_idx] = self.object_init_pos
        random_object_reset_quat = generate_random_quaternions(len(envs_idx), device=self.device)

        # 四元数のNaNチェック
        random_object_reset_quat = torch.nan_to_num(random_object_reset_quat, nan=0.0, posinf=0.0, neginf=0.0)
        # 四元数の正規化 (必須)
        random_object_reset_quat = random_object_reset_quat / (torch.norm(random_object_reset_quat, dim=1, keepdim=True) + 1e-6)

        self.object_init_quat = random_object_reset_quat
        self.object_quat[envs_idx] = self.object_init_quat
        self.object.set_pos(self.object_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.object.set_quat(self.object_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)

        # reset target
        self.target_pos[envs_idx] = self.target_init_pos
        random_target_reset_quat = generate_random_quaternions(len(envs_idx), device=self.device)

        # 四元数のNaNチェック
        random_target_reset_quat = torch.nan_to_num(random_target_reset_quat, nan=0.0, posinf=0.0, neginf=0.0)
        # 四元数の正規化 (必須)
        random_target_reset_quat = random_target_reset_quat / (torch.norm(random_target_reset_quat, dim=1, keepdim=True) + 1e-6)

        self.target_init_quat = random_target_reset_quat
        self.target_quat[envs_idx] = self.target_init_quat
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.set_quat(self.target_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset_target_pose(self, reset_envs_idx):
        # 新しいランダム四元数を生成
        rot_quat = generate_random_quaternions(len(reset_envs_idx), device=self.device)

        # NaNチェックと正規化
        rot_quat = torch.nan_to_num(rot_quat, nan=0.0, posinf=0.0, neginf=0.0)
        rot_quat = rot_quat / (torch.norm(rot_quat, dim=1, keepdim=True) + 1e-6)

        self.target.set_quat(rot_quat, zero_velocity=True, envs_idx=reset_envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_action_rate(self):
        # NaNを防ぐためにtorch.clampを使用
        diff = torch.clamp(self.last_actions - self.actions, -10.0, 10.0)
        return torch.sum(torch.square(diff), dim=1)

    def _reward_distance(self):
        # 距離値をクリップして巨大な値を防ぐ
        return torch.clamp(self.goal_dist, 0.0, 10.0)

    def _reward_rotation(self):
        # ゼロ除算を防ぐための安全策を追加
        eps = self.rot_eps + 1e-6
        rot_dist = torch.clamp(torch.abs(self.rot_dist), 0.0, 10.0)  # 極端な値の防止
        return 1.0 / (rot_dist + eps)

    def _reward_reach_goal_bonus(self):
        # NaNを防ぐためにtorch.whereの条件を安全に
        condition = torch.isfinite(self.rot_dist) & (torch.abs(self.rot_dist) <= self.success_tolerance)
        return torch.where(condition, self.reach_goal_bonus, torch.zeros_like(self.rot_dist))

    def _reward_fall_penalty(self):
        # NaNを防ぐためにtorch.whereの条件を安全に
        condition = torch.isfinite(self.goal_dist) & (self.goal_dist > self.fall_dist)
        return torch.where(condition, self.fall_penalty, torch.zeros_like(self.goal_dist))


    def process_actions(self, clip_actions: torch.Tensor):
        # NaNチェックを追加
        clip_actions = torch.nan_to_num(clip_actions, nan=0.0, posinf=0.0, neginf=0.0)

        # set position targets as moving average
        ema_actions = self.action_scale * clip_actions
        ema_actions += (1.0 - self.action_scale) * self.last_actions
        targets = ema_actions + self._prev_applied_actions

        # クランプする前にNaNチェック
        targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

        # clamp the targets
        lower_limits = self.all_dof_pos_limits[:, 0].repeat(self.num_envs, 1)
        upper_limits = self.all_dof_pos_limits[:, 1].repeat(self.num_envs, 1)

        # 制限値のNaNチェック
        lower_limits = torch.nan_to_num(lower_limits, nan=-1.0, posinf=-1.0, neginf=-1.0)
        upper_limits = torch.nan_to_num(upper_limits, nan=1.0, posinf=1.0, neginf=1.0)

        self._processed_actions[:] = torch.clamp(targets, lower_limits, upper_limits)

        # NaNチェック
        self._processed_actions[:] = torch.nan_to_num(self._processed_actions, nan=0.0, posinf=0.0, neginf=0.0)

        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]



def generate_random_quaternions(num_quats: int, device="cpu") -> torch.Tensor:
    """
    安全なランダム四元数を生成する。
    """
    u = torch.rand(num_quats, 3, device=device)

    # NaNを防ぐために0が入らないようにする
    u = torch.clamp(u, 1e-6, 1.0 - 1e-6)

    q = torch.stack([
        torch.sqrt(1 - u[:, 0]) * torch.sin(2 * np.pi * u[:, 1]),
        torch.sqrt(1 - u[:, 0]) * torch.cos(2 * np.pi * u[:, 1]),
        torch.sqrt(u[:, 0])     * torch.sin(2 * np.pi * u[:, 2]),
        torch.sqrt(u[:, 0])     * torch.cos(2 * np.pi * u[:, 2]),
    ], dim=-1)

    # 正規化して単位四元数にする
    q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-6)

    # 最終的なNaNチェック
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    return q

@torch.jit.script
def quaternion_difference(object_rot, target_rot):
    # NaNチェックを追加
    object_rot = torch.nan_to_num(object_rot, nan=0.0, posinf=0.0, neginf=0.0)
    target_rot = torch.nan_to_num(target_rot, nan=0.0, posinf=0.0, neginf=0.0)

    # 四元数の正規化
    object_rot = object_rot / (torch.norm(object_rot, dim=1, keepdim=True) + 1e-6)
    target_rot = target_rot / (torch.norm(target_rot, dim=1, keepdim=True) + 1e-6)

    # 共役四元数計算の前にもう一度NaNチェック
    target_rot_conj = quat_conjugate(target_rot)
    target_rot_conj = torch.nan_to_num(target_rot_conj, nan=0.0, posinf=0.0, neginf=0.0)

    result = quat_mul(object_rot, target_rot_conj)
    # 最終結果のNaNチェック
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # NaNチェックを追加
    object_rot = torch.nan_to_num(object_rot, nan=0.0, posinf=0.0, neginf=0.0)
    target_rot = torch.nan_to_num(target_rot, nan=0.0, posinf=0.0, neginf=0.0)

    # 四元数の正規化
    object_rot = object_rot / (torch.norm(object_rot, dim=1, keepdim=True) + 1e-6)
    target_rot = target_rot / (torch.norm(target_rot, dim=1, keepdim=True) + 1e-6)

    # 共役四元数計算
    rot_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_diff = torch.nan_to_num(rot_diff, nan=0.0, posinf=0.0, neginf=0.0)

    # ノルム計算と範囲制限
    norm_result = torch.norm(rot_diff[:, 1:4], p=2, dim=-1)
    norm_result = torch.clamp(norm_result, 0.0, 1.0)  # asinの入力範囲を確保

    result = 2.0 * torch.asin(norm_result)
    # 最終結果のNaNチェック
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result