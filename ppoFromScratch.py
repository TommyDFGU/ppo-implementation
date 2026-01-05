"""
PPO (Proximal Policy Optimization) Implementation from Scratch

This implementation follows the PPO algorithm as described in:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- CleanRL's PPO implementation (conceptual reference)

Key components:
1. Actor-Critic architecture with shared or separate networks
2. Generalized Advantage Estimation (GAE)
3. Clipped surrogate objective for policy updates
4. Value function clipping
5. Entropy regularization
6. Orthogonal initialization
7. Support for vectorized environments
8. Support for both discrete and continuous action spaces
"""

import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Environment
    env_id: str = "Pendulum-v1"  # Start with Inverted Pendulum
    num_envs: int = 4  # Number of parallel environments
    seed: int = 1
    
    # Training
    total_timesteps: int = 500000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    
    # PPO hyperparameters
    num_steps: int = 2048  # Rollout length (M in the paper)
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    num_minibatches: int = 32  # Number of minibatches per update
    update_epochs: int = 10  # Number of update epochs (K in the paper)
    clip_coef: float = 0.2  # Clipping coefficient (epsilon)
    ent_coef: float = 0.01  # Entropy coefficient (0.01 for continuous, 0.0 for discrete)
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    clip_vloss: bool = True  # Whether to clip value loss
    norm_adv: bool = True  # Whether to normalize advantages
    
    # Logging
    track: bool = False  # Use wandb
    capture_video: bool = False
    
    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Orthogonal initialization for neural network layers.
    Preserves gradient magnitudes and helps with training stability.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    The actor outputs action probabilities (discrete) or mean/std (continuous).
    The critic estimates the value function V(s).
    
    Design choice: Separate networks for actor and critic.
    Alternative: Shared feature extractor with separate heads.
    We use separate networks for simplicity and clarity.
    """
    
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        self.is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
        
        # Store action bounds for clipping (continuous actions only)
        if self.is_continuous:
            self.action_low = torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            self.action_high = torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        
        # Value function network (critic)
        # Outputs V(s) - the expected return from state s
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # std=1.0 for value head
        )
        
        # Policy network (actor)
        if self.is_continuous:
            # Continuous actions: output mean and log_std for Normal distribution
            action_dim = np.array(envs.single_action_space.shape).prod()
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, action_dim), std=0.01),  # Small std for policy head
            )
            # Learn log_std as a parameter (state-independent)
            # Alternative: output log_std from network (state-dependent)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # Discrete actions: output logits for Categorical distribution
            n_actions = envs.single_action_space.n
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, n_actions), std=0.01),
            )
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate V(s) for given observations"""
        return self.critic(obs)
    
    def get_action_and_value(
        self, 
        obs: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value for given observations.
        
        Returns:
            action: Sampled action (or provided action if given)
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: Value estimate V(s)
        """
        value = self.get_value(obs)
        
        if self.is_continuous:
            mean = self.actor_mean(obs)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            dist = Normal(mean, std)
        else:
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        # Don't clip actions here - it breaks log_prob computation
        # Actions are clipped when executing in the environment
        log_prob = dist.log_prob(action)
        
        # For continuous actions, log_prob has shape (batch, action_dim)
        # We sum over action dimensions to get scalar log_prob per sample
        if self.is_continuous:
            log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy()
        if self.is_continuous:
            entropy = entropy.sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE combines multiple n-step returns with exponential weighting:
    A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
    
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    
    Reduces variance compared to single-step advantages and balances bias vs variance.
    
    Args:
        rewards: Shape (num_steps, num_envs)
        values: Shape (num_steps, num_envs)
        dones: Shape (num_steps, num_envs)
        next_value: Shape (1, num_envs) - value of next state after rollout
        gamma: Discount factor
        gae_lambda: GAE parameter (0=TD, 1=MC)
    
    Returns:
        advantages: Shape (num_steps, num_envs)
        returns: Shape (num_steps, num_envs) - advantages + values
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    
    # Compute advantages backwards in time
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # Last step: bootstrap from next_value
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            # Intermediate steps: bootstrap from next step's value
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        
        # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        
        # GAE: A_t = delta_t + (gamma * lambda) * A_{t+1}
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    
    # Returns = advantages + values (used for value function learning)
    returns = advantages + values
    
    return advantages, returns


def ppo_update(
    agent: ActorCritic,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    config: PPOConfig,
) -> dict:
    """
    Perform PPO policy and value function updates.
    
    Uses clipped surrogate objective to prevent large policy updates:
    L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
    
    where r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) is the importance ratio.
    
    Args:
        agent: Actor-critic network
        optimizer: Optimizer for network parameters
        obs: Observations (batch_size, obs_dim)
        actions: Actions taken (batch_size, action_dim or batch_size)
        logprobs: Old log probabilities (batch_size)
        advantages: Advantage estimates (batch_size)
        returns: Return estimates (batch_size)
        values: Old value estimates (batch_size)
        config: PPO configuration
    
    Returns:
        Dictionary with loss statistics
    """
    # Flatten batch for minibatch updates
    batch_size = config.batch_size
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_actions = actions.reshape((-1,) + actions.shape[2:])
    b_logprobs = logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    
    # Indices for minibatch sampling
    b_inds = np.arange(batch_size)
    clipfracs = []
    
    # Multiple epochs of updates on the same data
    for epoch in range(config.update_epochs):
        np.random.shuffle(b_inds)
        
        # Process in minibatches
        for start in range(0, batch_size, config.minibatch_size):
            end = start + config.minibatch_size
            mb_inds = b_inds[start:end]
            
            # Get new policy predictions
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            
            # Importance ratio: r_t = pi_new(a|s) / pi_old(a|s)
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            
            # Approximate KL divergence for monitoring
            # KL(pi_old || pi_new) ≈ E[(ratio - 1) - log(ratio)]
            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]
            
            # Normalize advantages (standard practice in PPO)
            # Reduces variance and improves stability
            mb_advantages = b_advantages[mb_inds]
            if config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Policy loss: clipped surrogate objective
            # L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
            # We negate because we want to maximize, but optimizers minimize
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - config.clip_coef, 1 + config.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # Value loss
            newvalue = newvalue.view(-1)
            if config.clip_vloss:
                # Clipped value loss for additional stability
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                # Standard MSE value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
            
            # Entropy loss: encourages exploration
            # Higher entropy = more random actions = more exploration
            entropy_loss = entropy.mean()
            
            # Total loss: policy loss - entropy bonus + value loss
            # We subtract entropy because we want to maximize it (but optimizers minimize)
            loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
            
            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: prevents exploding gradients
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()
    
    # Compute explained variance (diagnostic metric)
    # Measures how well value function predicts returns
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    
    return {
        "pg_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": np.mean(clipfracs),
        "explained_var": explained_var,
    }


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    """Create environment with optional video recording"""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def train_ppo(config: PPOConfig):
    """
    Main training loop for PPO.
    
    PPO training consists of:
    1. Rollout collection: collect trajectories using current policy
    2. Advantage computation: compute GAE advantages
    3. Optimization: update policy and value function with clipped objective
    """
    # Compute derived hyperparameters
    config.batch_size = int(config.num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = config.total_timesteps // config.batch_size
    
    # Setup logging
    run_name = f"{config.env_id}__ppo__{config.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )
    
    # Seeding for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create vectorized environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, i, config.capture_video, run_name) for i in range(config.num_envs)]
    )
    
    # Create agent
    agent = ActorCritic(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    
    # Storage for rollout data
    # Shape: (num_steps, num_envs, ...)
    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    
    # Initialize environment
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    
    print(f"Starting training on {config.env_id}")
    print(f"Total iterations: {config.num_iterations}")
    print(f"Batch size: {config.batch_size}, Minibatch size: {config.minibatch_size}")
    
    # Main training loop
    for iteration in range(1, config.num_iterations + 1):
        # Learning rate annealing
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # ========== ROLLOUT COLLECTION ==========
        # Collect trajectories using current policy
        for step in range(0, config.num_steps):
            global_step += config.num_envs
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
                    envs.single_action_space.low,
                    envs.single_action_space.high
                )
            
            next_obs, reward, terminations, truncations, infos = envs.step(action_np)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # Log episode statistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        # ========== ADVANTAGE COMPUTATION ==========
        # Compute GAE advantages using collected rollout
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_gae(
                rewards, values, dones, next_value, config.gamma, config.gae_lambda
            )
        
        # ========== OPTIMIZATION ==========
        # Update policy and value function
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
    import os
    os.makedirs("models", exist_ok=True)
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'config': config,
        'env_id': config.env_id,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    envs.close()
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    # Example: Train on Inverted Pendulum
    # Better hyperparameters for Pendulum-v1 (continuous control)
    config = PPOConfig(
        env_id="Pendulum-v1",
        num_envs=8,
        total_timesteps=3000000,
        learning_rate=2.5e-4,
        num_steps=2048,  # Rollout length
        update_epochs=10,  # Number of update epochs
        ent_coef=0.01,  # Small entropy bonus for exploration
        clip_coef=0.2,  # Standard clipping
        num_minibatches=32,  # Minibatches
    )
    
    train_ppo(config)

