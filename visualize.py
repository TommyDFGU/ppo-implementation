"""
Generate GIFs from trained PPO models

This script loads a trained model and generates GIFs showing the agent's behavior.
Perfect for creating visualizations for reports and presentations.

Usage:
    # Generate GIF from trained model
    python3 visualize.py --model_path models/Pendulum-v1__ppo__1__1234567890_final.pth --output gifs/pendulum.gif
    
    # With custom episode length
    python3 visualize.py --model_path models/Pendulum-v1__ppo__1__1234567890_final.pth --output gifs/pendulum.gif --max_steps 500
"""

import argparse
import os
import torch
import numpy as np
import gymnasium as gym
from ppoFromScratch import ActorCritic, PPOConfig

try:
    import imageio
except ImportError:
    print("Warning: imageio not installed. Install it with: pip install imageio")
    imageio = None


def create_gif(frames, output_path, fps=30):
    """
    Create a GIF from a list of frames.
    
    Args:
        frames: List of RGB frames (numpy arrays)
        output_path: Path to save the GIF
        fps: Frames per second for the GIF
    """
    if imageio is None:
        raise ImportError("imageio is required. Install with: pip install imageio")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Create GIF with infinite looping (loop=0 means infinite loop)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {output_path} (infinite loop enabled)")


def visualize_gymnasium(model_path, output_path, max_steps=1000, fps=30, seed=0):
    """
    Visualize a trained model on a Gymnasium environment.
    
    Args:
        model_path: Path to saved model
        output_path: Path to save GIF
        max_steps: Maximum steps per episode
        fps: Frames per second for GIF
        seed: Random seed
    """
    print(f"Loading model from {model_path}...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    env_id = checkpoint['env_id']
    
    print(f"Environment: {env_id}")
    
    # Create environment with rendering
    env = gym.make(env_id, render_mode='rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Create agent
    # We need to create a dummy vectorized env to get the spaces
    dummy_envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(1)])
    agent = ActorCritic(dummy_envs)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    dummy_envs.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    
    # Run episode and collect frames
    print("Running episode and collecting frames...")
    frames = []
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    steps = 0
    
    with torch.no_grad():
        while steps < max_steps:
            # Get action from policy
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action_np = action.cpu().numpy()[0]
            
            # Clip action if continuous
            if agent.is_continuous:
                action_np = np.clip(
                    action_np,
                    env.action_space.low,
                    env.action_space.high
                )
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Render frame
            frame = env.render()
            frames.append(frame)
            
            if done:
                print(f"Episode finished after {steps} steps, total reward: {total_reward:.2f}")
                break
    
    env.close()
    
    # Create GIF
    if len(frames) > 0:
        create_gif(frames, output_path, fps=fps)
        print(f"✅ Created GIF with {len(frames)} frames")
    else:
        print("❌ No frames collected!")


def list_models():
    """List all available trained models."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found. Train a model first!")
        return
    
    models = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not models:
        print("No trained models found in models/ directory.")
        return
    
    print("Available trained models:")
    for i, model in enumerate(models, 1):
        model_path = os.path.join(models_dir, model)
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            env_name = checkpoint.get('env_name') or checkpoint.get('env_id', 'unknown')
            print(f"  {i}. {model}")
            print(f"     Environment: {env_name}")
            print(f"     Path: {model_path}")
            print()
        except Exception as e:
            print(f"  {i}. {model} (error loading: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GIFs from trained PPO models")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model (.pth file). Use --list to see available models.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available trained models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gifs/agent.gif",
        help="Output path for GIF",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for GIF",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        exit(0)
    
    if args.model_path is None:
        print("Error: --model_path is required (or use --list to see available models)")
        list_models()
        exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("Available models in models/ directory:")
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith(".pth"):
                    print(f"  - models/{f}")
        exit(1)
    
    visualize_gymnasium(args.model_path, args.output, args.max_steps, args.fps, args.seed)

