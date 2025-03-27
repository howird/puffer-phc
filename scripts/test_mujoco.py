import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import yaml

from puffer_phc import ASSET_DIR
from puffer_phc.humanoid_phc_mujoco import HumanoidPHCMujoco


def load_cfg(cfg_path: str) -> Dict:
    """Load config from yaml file"""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    # Load config
    cfg = load_cfg(str(ASSET_DIR / "cfg" / "phc_mujoco.yaml"))
    
    # Create environment
    env = HumanoidPHCMujoco(
        cfg=cfg,
        sim_params=None,
        device_type="cuda",
        device_id=0,
        headless=False,
    )

    print("\nEnvironment created successfully!")
    print(f"Observation space: {env.single_observation_space}")
    print(f"Action space: {env.single_action_space}")
    print(f"Number of environments: {env.num_envs}")

    # Test basic functionality
    print("\nTesting reset...")
    obs = env.reset()
    print("Reset successful!")
    print(f"Observation shape: {obs.shape}")

    print("\nTesting random actions...")
    num_steps = 100
    start_time = time.time()

    for i in range(num_steps):
        # Random actions
        actions = torch.rand(
            (env.num_envs, env.num_actions), 
            device=env.device
        ) * 2 - 1  # Scale to [-1, 1]
        
        obs, reward, done, info = env.step(actions)
        
        # Print progress
        if i % 10 == 0:
            print(f"Step {i}, Reward: {reward.mean().item():.3f}")
        
        # Render
        env.render()
        
        # Optional: sleep to slow down visualization
        time.sleep(0.01)

    end_time = time.time()
    fps = num_steps / (end_time - start_time)
    print(f"\nFinished {num_steps} steps at {fps:.1f} FPS")

    # Clean up
    env.close()
    print("\nEnvironment closed successfully!")


if __name__ == "__main__":
    main()
