import argparse
import os
import pickle
import shutil

from shadow_hand_env import ShadowHandEnv
from rsl_rl.runners import OnPolicyRunner

import os
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_dofs": 24,
        "num_actions": 24,
        # joint/link names
        "default_joint_angles": {  # [rad]
            'lh_WRJ2': 0.0,
            'lh_WRJ1': 0.0,
            'lh_FFJ4': 0.0,
            'lh_FFJ3': 0.0,
            'lh_FFJ2': 0.0,
            'lh_FFJ1': 0.0,
            'lh_MFJ4': 0.0,
            'lh_MFJ3': 0.0,
            'lh_MFJ2': 0.0,
            'lh_MFJ1': 0.0,
            'lh_RFJ4': 0.0,
            'lh_RFJ3': 0.0,
            'lh_RFJ2': 0.0,
            'lh_RFJ1': 0.0,
            'lh_LFJ5': 0.0,
            'lh_LFJ4': 0.0,
            'lh_LFJ3': 0.0,
            'lh_LFJ2': 0.0,
            'lh_LFJ1': 0.0,
            'lh_THJ5': 0.0,
            'lh_THJ4': 0.0,
            'lh_THJ3': 0.0,
            'lh_THJ2': 0.0,
            'lh_THJ1': 0.0,
        },

        "all_dof_names": [
            'lh_WRJ2',
            'lh_WRJ1',
            'lh_FFJ4',
            'lh_FFJ3',
            'lh_FFJ2',
            'lh_FFJ1',
            'lh_MFJ4',
            'lh_MFJ3',
            'lh_MFJ2',
            'lh_MFJ1',
            'lh_RFJ4',
            'lh_RFJ3',
            'lh_RFJ2',
            'lh_RFJ1',
            'lh_LFJ5',
            'lh_LFJ4',
            'lh_LFJ3',
            'lh_LFJ2',
            'lh_LFJ1',
            'lh_THJ5',
            'lh_THJ4',
            'lh_THJ3',
            'lh_THJ2',
            'lh_THJ1',
        ],
        # genesisにはmimic jointを反映する機能がないので、現状は全ての軸を動作させている
        "action_dof_names": [
            'lh_WRJ2',
            'lh_WRJ1',
            'lh_FFJ4',
            'lh_FFJ3',
            'lh_FFJ2',
            'lh_FFJ1',
            'lh_MFJ4',
            'lh_MFJ3',
            'lh_MFJ2',
            'lh_MFJ1',
            'lh_RFJ4',
            'lh_RFJ3',
            'lh_RFJ2',
            'lh_RFJ1',
            'lh_LFJ5',
            'lh_LFJ4',
            'lh_LFJ3',
            'lh_LFJ2',
            'lh_LFJ1',
            'lh_THJ5',
            'lh_THJ4',
            'lh_THJ3',
            'lh_THJ2',
            'lh_THJ1',
        ],
        # PD
        "kp": 3.0,
        "kd": 0.3,
        # hand pose
        "hand_init_pos": [-0.2, 0.0, 0.375],
        "hand_init_quat": [0, 0.7660446512928001, 0, 0.6427878401730001],
        "episode_length_s": 20.0,
        "action_scale": 0.8,
        "simulate_action_latency": True,
        "clip_actions": 1,
    }
    obs_cfg = {
        "num_obs": 66,
        "num_privileged_obs": 96,
        "obs_scales": {
            "dof_pos": 1.0,
            "object_pos": 1.0,
            "object_quat": 1.0,
            "target_pos": 1.0,
            "target_quat": 1.0,
            "quat_diff": 1.0,
            "last_actions": 1.0,
        },
    }
    reward_cfg = {
        "rot_eps": 0.1,
        "success_tolerance": 0.25,
        "reach_goal_bonus":  250,
        "fall_dist": 0.35,
        "fall_penalty": 0,
        "reward_scales": {
            "distance": 0,
            "rotation": 1.0,
            "action_rate": -0.001,
            "reach_goal_bonus": 1.0,
            "fall_penalty": -1.0,
        },
    }
    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="shadow-rotate")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--max_iterations", type=int, default=20000)
    args = parser.parse_args()

    # gs.init(logging_level="warning")
    gs.init(backend=gs.cuda)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    env = ShadowHandEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/manipulation/shadow_hand_train.py
"""
