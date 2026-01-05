"""
Train PPO on BRAX environments (Ant, Humanoid, etc.)

This script demonstrates how to use the PPO implementation with BRAX environments.
BRAX provides highly vectorized, GPU-accelerated physics simulations.

Usage:
    python3 train_brax.py --env_name ant --num_envs 256
    python3 train_brax.py --env_name humanoid --num_envs 128
"""

import argparse
import random
import time
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ppoFromScratch import (
    PPOConfig,
    ActorCritic,
    compute_gae,
    ppo_update,
)
from brax_wrapper import make_brax_env, BRAX_ENVIRONMENTS


def train_ppo_with_env(envs, config: PPOConfig, env_name: str):
    """
    Train PPO with a pre-created environment (for BRAX compatibility).
    
    Similar to train_ppo but accepts a pre-created environment instead of creating
    one from env_id. Needed for BRAX environments which require special initialization.
    
    Args:
        envs: Pre-created Gymnasium environment (can be vectorized)
        config: PPO configuration
        env_name: Environment name (for logging)
    """
    # Compute derived hyperparameters
    # For BRAX, num_envs is handled internally, so we use the actual batch size
    # BRAX environments return batched observations/actions
    if hasattr(envs, 'num_envs'):
        actual_num_envs = envs.num_envs
    elif hasattr(envs, 'batch_size'):
        actual_num_envs = envs.batch_size
    elif hasattr(envs, 'env') and hasattr(envs.env, 'batch_size'):
        # BRAX VectorGymWrapper wraps an env with batch_size
        actual_num_envs = envs.env.batch_size
    else:
        # Try to infer from observation shape
        # BRAX vectorized envs have shape (batch_size, obs_dim)
        obs_shape = envs.observation_space.shape
        if len(obs_shape) > 1 and obs_shape[0] > 1:
            actual_num_envs = obs_shape[0]
        else:
            # Single environment
            actual_num_envs = 1
    
    config.batch_size = int(actual_num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = config.total_timesteps // config.batch_size
    
    # Setup logging
    run_name = f"{env_name}__ppo__{config.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )
    
    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Actual number of parallel environments: {actual_num_envs}")
    
    # Create agent
    agent = ActorCritic(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    
    # Get single environment spaces (for storage shape)
    # BRAX VectorGymWrapper provides single_observation_space and single_action_space
    if hasattr(envs, 'single_observation_space'):
        single_obs_space = envs.single_observation_space
        single_action_space = envs.single_action_space
    else:
        # Fallback: use observation/action space directly
        single_obs_space = envs.observation_space
        single_action_space = envs.action_space
    
    # Storage for rollout data
    # Shape: (num_steps, num_envs, ...)
    obs = torch.zeros((config.num_steps, actual_num_envs) + single_obs_space.shape).to(device)
    actions = torch.zeros((config.num_steps, actual_num_envs) + single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, actual_num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, actual_num_envs)).to(device)
    dones = torch.zeros((config.num_steps, actual_num_envs)).to(device)
    values = torch.zeros((config.num_steps, actual_num_envs)).to(device)
    
    # Initialize environment
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    # Copy array to make it writable (fixes PyTorch warning)
    next_obs = torch.Tensor(np.array(next_obs, copy=True)).to(device)
    next_done = torch.zeros(actual_num_envs).to(device)
    
    print(f"Starting training on {env_name}")
    print(f"Total iterations: {config.num_iterations}")
    print(f"Batch size: {config.batch_size}, Minibatch size: {config.minibatch_size}\n")
    
    # Main training loop
    for iteration in range(1, config.num_iterations + 1):
        # Learning rate annealing
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # ========== ROLLOUT COLLECTION ==========
        for step in range(0, config.num_steps):
            global_step += actual_num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get action from current policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            # Execute action in environment
            action_np = action.cpu().numpy()
            
            # Clip continuous actions to environment bounds
            if agent.is_continuous:
                action_np = np.clip(
                    action_np,
                    single_action_space.low,
                    single_action_space.high
                )
            
            next_obs, reward, terminations, truncations, infos = envs.step(action_np)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # Copy arrays to make them writable (fixes PyTorch warning)
            next_obs = torch.Tensor(np.array(next_obs, copy=True)).to(device)
            next_done = torch.Tensor(np.array(next_done, copy=True)).to(device)
            
            # Log episode statistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        # ========== ADVANTAGE COMPUTATION ==========
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_gae(
                rewards, values, dones, next_value, config.gamma, config.gae_lambda
            )
        
        # ========== OPTIMIZATION ==========
        stats = ppo_update(
            agent, optimizer, obs, actions, logprobs, advantages, returns, values, config
        )
        
        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", stats["v_loss"], global_step)
        writer.add_scalar("losses/policy_loss", stats["pg_loss"], global_step)
        writer.add_scalar("losses/entropy", stats["entropy"], global_step)
        writer.add_scalar("losses/old_approx_kl", stats["old_approx_kl"], global_step)
        writer.add_scalar("losses/approx_kl", stats["approx_kl"], global_step)
        writer.add_scalar("losses/clipfrac", stats["clipfrac"], global_step)
        writer.add_scalar("losses/explained_variance", stats["explained_var"], global_step)
        
        # Steps per second
        sps = int(global_step / (time.time() - start_time))
        
        # Diagnostic print every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{config.num_iterations}, SPS: {sps}, "
                  f"Policy Loss: {stats['pg_loss']:.4f}, Value Loss: {stats['v_loss']:.4f}, "
                  f"KL: {stats['approx_kl']:.4f}, ClipFrac: {stats['clipfrac']:.3f}")
        else:
            print(f"Iteration {iteration}/{config.num_iterations}, SPS: {sps}")
        
        writer.add_scalar("charts/SPS", sps, global_step)
    
    # Save trained model
    model_path = f"models/{run_name}_final.pth"
    os.makedirs("models", exist_ok=True)
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'config': config,
        'env_name': env_name,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    envs.close()
    writer.close()
    print("Training complete!")


def train_ppo_brax(env_name: str, num_envs: int = 64, total_timesteps: int = 1_000_000, 
                   learning_rate: float = 3e-4, seed: int = 1):
    """
    Train PPO on a BRAX environment.
    
    BRAX environments are highly vectorized and run efficiently on GPUs.
    We use BRAX's native vectorization rather than SyncVectorEnv for better performance.
    
    Args:
        env_name: BRAX environment name (e.g., "ant", "humanoid")
        num_envs: Number of parallel environments (BRAX handles this efficiently)
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        seed: Random seed
    """
    if env_name not in BRAX_ENVIRONMENTS:
        print(f"Warning: {env_name} not in known BRAX environments.")
        print(f"Available: {BRAX_ENVIRONMENTS}")
    
    # Create BRAX environment (already vectorized)
    print(f"Creating BRAX environment: {env_name} with {num_envs} parallel environments...")
    envs = make_brax_env(env_name, num_envs=num_envs, seed=seed)
    
    # Get environment info
    obs_shape = envs.observation_space.shape
    action_shape = envs.action_space.shape
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    print(f"Action space: {envs.action_space}")
    
    # BRAX-specific hyperparameters
    # BRAX environments typically need:
    # - Longer rollouts (more steps per update)
    # - More update epochs (reuse data more)
    # - Higher learning rates sometimes
    
    config = PPOConfig(
        env_id=env_name,  # Used for logging only
        num_envs=1,  # BRAX handles vectorization internally, so we set this to 1
        seed=seed,
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        num_steps=2048,  # Rollout length
        update_epochs=10,  # Number of update epochs
        num_minibatches=32,  # Minibatches
        clip_coef=0.2,  # Clipping coefficient
        ent_coef=0.01,  # Entropy coefficient (helps exploration)
        vf_coef=0.5,  # Value function coefficient
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        max_grad_norm=0.5,  # Gradient clipping
        clip_vloss=True,  # Value clipping
        norm_adv=True,  # Normalize advantages
        anneal_lr=True,  # Learning rate annealing
    )
    
    print(f"\nStarting PPO training on {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {num_envs}")
    print(f"Rollout length: {config.num_steps}")
    print(f"Update epochs: {config.update_epochs}\n")
    
    train_ppo_with_env(envs, config, env_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on BRAX environments")
    parser.add_argument(
        "--env_name",
        type=str,
        default="ant",
        choices=BRAX_ENVIRONMENTS,
        help="BRAX environment name",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=256,
        help="Number of parallel environments (BRAX handles vectorization)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,  # Reduced for demonstration (shows learning, not SOTA)
        help="Total training timesteps (1M is sufficient to demonstrate learning)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    train_ppo_brax(
        env_name=args.env_name,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

