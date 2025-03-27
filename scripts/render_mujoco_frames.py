import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml

from puffer_phc import ASSET_DIR
from puffer_phc.humanoid_phc_mujoco import HumanoidPHCMujoco


def load_cfg(cfg_path: str):
    """Load config from yaml file"""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    # Create output directory
    output_dir = Path("output_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Load config
    cfg = load_cfg(str(ASSET_DIR / "cfg" / "phc_mujoco.yaml"))
    
    # Create environment
    env = HumanoidPHCMujoco(
        cfg=cfg,
        sim_params=None,
        device_type="cuda",
        device_id=0,
        headless=True,  # Use headless since we're rendering to images
    )

    print("\nEnvironment created successfully!")
    
    # Reset environment
    obs = env.reset()
    
    # Run simulation and save frames
    num_steps = 100
    for i in range(num_steps):
        # Random actions
        actions = torch.rand(
            (env.num_envs, env.num_actions), 
            device=env.device
        ) * 2 - 1  # Scale to [-1, 1]
        
        # Step simulation
        obs, reward, done, info = env.step(actions)
        
        # Render and save frame
        frame = env.mujoco_base.render_to_image(width=1280, height=720)
        cv2.imwrite(
            str(output_dir / f"frame_{i:04d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )
        
        if i % 10 == 0:
            print(f"Rendered frame {i}")

    print(f"\nRendered {num_steps} frames to {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
