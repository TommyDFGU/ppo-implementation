# PPO from Scratch: Implementation and Analysis

This project implements Proximal Policy Optimization (PPO) from scratch in PyTorch, with support for vectorized environments and both discrete and continuous action spaces. The implementation follows best practices from CleanRL and the original PPO paper.

## Project Overview

This implementation includes:
- **PPO algorithm** with clipped surrogate objective
- **Generalized Advantage Estimation (GAE)** for variance reduction
- **Vectorized environment support** for efficient parallel data collection
- **Continuous and discrete action spaces** (tested on Pendulum-v1 and CartPole-v1)
- **Value function clipping** for additional stability
- **Entropy regularization** for exploration
- **Orthogonal initialization** for stable training
- **Comprehensive logging** with TensorBoard

## Requirements

```bash
pip install torch gymnasium numpy tensorboard
```

For BRAX environments (optional):
```bash
pip install brax
```

## Quick Start

### Basic Usage

Train PPO on Inverted Pendulum (continuous control):
```bash
python3 ppoFromScratch.py
```

### Custom Configuration

Modify the `PPOConfig` in `ppoFromScratch.py` or create a new script:

```python
from ppoFromScratch import PPOConfig, train_ppo

config = PPOConfig(
    env_id="Pendulum-v1",      # Environment
    num_envs=4,                # Number of parallel environments
    total_timesteps=500000,    # Total training steps
    learning_rate=3e-4,        # Learning rate
    num_steps=2048,            # Rollout length
    update_epochs=10,         # Number of update epochs
    clip_coef=0.2,             # Clipping coefficient
)

train_ppo(config)
```

### Example: CartPole (Discrete Actions)

```python
config = PPOConfig(
    env_id="CartPole-v1",
    num_envs=4,
    total_timesteps=250000,
    num_steps=128,
    update_epochs=4,
)
train_ppo(config)
```

## Monitoring Training

Training logs are saved to `runs/` directory. View with TensorBoard:

```bash
tensorboard --logdir runs
```

Key metrics logged:
- `charts/episodic_return`: Episode returns (learning curve)
- `losses/policy_loss`: Policy gradient loss
- `losses/value_loss`: Value function loss
- `losses/entropy`: Policy entropy (exploration)
- `losses/approx_kl`: Approximate KL divergence
- `losses/clipfrac`: Fraction of clipped updates
- `losses/explained_variance`: Value function quality

## Project Structure

```
code/
├── ppoFromScratch.py    # Main PPO implementation
├── README.md            # This file
├── REPORT.md            # Theoretical report
└── runs/                # TensorBoard logs
```

## Key Components

### 1. Actor-Critic Network (`ActorCritic`)
- Separate networks for policy (actor) and value (critic)
- Supports both discrete (Categorical) and continuous (Normal) action distributions
- Orthogonal initialization for stability

### 2. GAE Computation (`compute_gae`)
- Implements Generalized Advantage Estimation
- Balances bias (TD) vs variance (Monte Carlo)
- Standard in modern policy gradient methods

### 3. PPO Update (`ppo_update`)
- Clipped surrogate objective
- Value function clipping (optional)
- Entropy regularization
- Multiple epochs of updates on same data

### 4. Training Loop (`train_ppo`)
- Rollout collection phase
- Advantage computation phase
- Optimization phase
- Vectorized environment support

## Results

### Pendulum-v1 (Continuous Control)
- **Training**: 3,000,000 timesteps with 8 parallel environments
- **Final Performance**: Mean episodic return improved from ~-1600 (random) to >-200 (trained)
- **Key Metrics**: 
  - KL divergence remained stable (< 0.1)
  - Explained variance > 0.5 (value function learning well)
  - Clip fraction moderate (0.1-0.3)
- **Visualization**: See `gifs/pendulum2.gif` for trained agent behavior

### Ant (BRAX Environment)
- **Training**: 1,000,000 timesteps with 64 parallel environments
- **Final Performance**: Agent successfully learns locomotion behavior
- **Key Observations**:
  - Episodic return increases steadily over training
  - Agent learns to coordinate multiple joints for forward movement
  - Training demonstrates PPO effectiveness on complex high-dimensional control task


Tested environments:
- ✅ `Pendulum-v1` (continuous)
- ✅ `CartPole-v1` (discrete)
- ✅ `MountainCar-v0` (discrete)

## References

- **PPO Paper**: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- **CleanRL**: https://docs.cleanrl.dev/rl-algorithms/ppo/
- **Implementation Details**: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

## 🚀 Training on BRAX Environments

BRAX provides highly vectorized, GPU-accelerated physics simulations. The implementation includes full BRAX support for complex environments like Ant and Humanoid.

### Installation

```bash
pip install brax jax jaxlib
```

### Quick Start

Train on Ant environment (demonstration):
```bash
python3 train_brax.py --env_name ant --num_envs 64 --total_timesteps 1_000_000
```
- **Time**: ~2-3 hours on 2020 Intel MacBook Air, ~5-10 minutes on GPU
- **Shows learning**: ✅ Yes - episodic return increases over time

Train on Humanoid environment (demonstration):
```bash
python3 train_brax.py --env_name humanoid --num_envs 32 --total_timesteps 2_000_000
```
- **Time**: ~4-6 hours on 2020 Intel MacBook Air, ~10-20 minutes on GPU
- **Shows learning**: ✅ Yes - demonstrates PPO works on complex tasks

### Available BRAX Environments

- `ant` - Ant locomotion (recommended starting point)
- `humanoid` - Humanoid locomotion (more complex)
- `halfcheetah` - HalfCheetah locomotion
- `hopper` - Hopper locomotion
- `walker2d` - Walker2D locomotion
- `fetch`, `grasp`, `ur5e` - Manipulation tasks
- `reacher`, `acrobot` - Control tasks

### BRAX-Specific Considerations

**For Demonstration (Recommended for Academic Projects)**:
1. **Reduced timesteps**: 1-2M is sufficient to show learning
   - Ant: 1M timesteps (~2-3 hours on CPU)
   - Humanoid: 2M timesteps (~4-6 hours on CPU)
2. **Fewer parallel environments**: 32-64 for Ant, 16-32 for Humanoid
3. **Shorter rollouts**: `num_steps=1024` instead of `2048`

**For Full Training (Optional)**:
1. **Vectorization**: BRAX environments are natively vectorized and run efficiently on GPUs/TPUs
2. **Hyperparameters**: BRAX environments typically need:
   - Longer training (10M+ timesteps for complex tasks)
   - More parallel environments (128-512)
   - Standard PPO hyperparameters work well
3. **Performance**: BRAX can simulate thousands of environments in parallel on GPU

### Example Usage

**For Demonstration**:
```python
from train_brax import train_ppo_brax

# Train on Ant (shows learning in ~2-3 hours on CPU)
train_ppo_brax(
    env_name="ant",
    num_envs=64,
    total_timesteps=1_000_000,  # 1M is enough to demonstrate learning
    learning_rate=3e-4,
)

# Train on Humanoid (shows learning in ~4-6 hours on CPU)
train_ppo_brax(
    env_name="humanoid",
    num_envs=32,
    total_timesteps=2_000_000,  # 2M shows learning on complex task
    learning_rate=3e-4,
)
```

**For Full Training (Optional)**:
```python
# Train on Ant (full training for better performance)
train_ppo_brax(
    env_name="ant",
    num_envs=256,
    total_timesteps=10_000_000,  # 10M for SOTA performance
    learning_rate=3e-4,
)
```

### Monitoring BRAX Training

Use TensorBoard as usual:
```bash
tensorboard --logdir runs
```

Key metrics to watch:
- `episodic_return`: Should steadily increase
- `SPS` (Steps Per Second): Should be very high with BRAX (10k+)
- `policy_loss`, `value_loss`: Standard PPO diagnostics

## License

This is an academic project for educational purposes.

## Author

Tommy Glanan (M2 MED TSE)

