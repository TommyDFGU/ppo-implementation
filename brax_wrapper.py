"""
BRAX Environment Wrapper for PyTorch PPO

This module provides wrappers to use BRAX (JAX-based) environments with PyTorch.
BRAX environments are highly vectorized and run efficiently on GPUs/TPUs.

Key differences from standard Gymnasium environments:
- BRAX uses JAX arrays (need conversion to NumPy/PyTorch)
- BRAX environments are already vectorized (no need for SyncVectorEnv)
- BRAX provides Gymnasium-compatible wrappers via brax.io.gym
"""

import gymnasium as gym
import numpy as np
from typing import Optional


class BraxToGymnasiumWrapper:
    """
    Custom wrapper to convert BRAX environments to Gymnasium-compatible interface.
    
    This wrapper handles:
    - JAX array to NumPy conversion
    - Gymnasium API compatibility
    - Vectorized environment support
    """
    
    def __init__(self, brax_env, seed: int = 0):
        """
        Initialize the wrapper.
        
        Args:
            brax_env: BRAX environment instance
            seed: Random seed
        """
        import jax
        self.brax_env = brax_env
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        
        # Get observation and action spaces from BRAX environment
        obs_shape = brax_env.observation_size
        action_shape = brax_env.action_size
        
        # Create Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float32
        )
        
        # BRAX actions are typically in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_shape,),
            dtype=np.float32
        )
        
        # Initialize state
        self._state = None
        self.batch_size = getattr(brax_env, 'batch_size', 1)
        
        # Reset to initialize
        self.reset(seed=seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        import jax
        if seed is not None:
            self.seed = seed
            self.rng = jax.random.PRNGKey(seed)
        
        # Reset BRAX environment
        self.rng, reset_rng = jax.random.split(self.rng)
        state = self.brax_env.reset(reset_rng)
        self._state = state
        
        # Convert JAX array to NumPy
        obs = np.asarray(state.obs, dtype=np.float32)
        
        # For vectorized envs, we need to handle the batch dimension
        # But for compatibility, we'll return the full batch
        return obs, {}
    
    def step(self, action: np.ndarray):
        """Step the environment."""
        import jax.numpy as jnp
        
        # Convert NumPy to JAX
        action = np.array(action)  # Ensure it's a numpy array
        
        # BRAX expects actions to match the batch dimension of the state
        # Handle different action shapes
        if len(action.shape) == 1:
            # Single action: (action_dim,) -> (batch_size, action_dim)
            if self.batch_size > 1:
                # For vectorized env, broadcast single action to all envs
                action = np.tile(action[np.newaxis, :], (self.batch_size, 1))
            else:
                # Single env: add batch dimension
                action = action[np.newaxis, :]
        elif len(action.shape) == 2:
            # Already batched: (batch_size, action_dim)
            # Verify batch size matches
            if action.shape[0] != self.batch_size:
                raise ValueError(f"Action batch size {action.shape[0]} doesn't match environment batch size {self.batch_size}")
        
        action_jax = jnp.array(action)
        
        # Step BRAX environment
        self._state = self.brax_env.step(self._state, action_jax)
        
        # Extract observation, reward, done
        obs = np.asarray(self._state.obs, dtype=np.float32)
        reward = np.asarray(self._state.reward, dtype=np.float32)
        done = np.asarray(self._state.done, dtype=bool)
        
        # Handle vectorized vs single environment
        if self.batch_size > 1:
            # Vectorized environment: return batched arrays
            # Ensure reward and done are 1D arrays
            if len(reward.shape) == 0:
                reward = np.array([reward])
            if len(done.shape) == 0:
                done = np.array([done])
            # Flatten if needed
            reward = reward.flatten()
            done = done.flatten()
            truncation = np.zeros_like(done, dtype=bool)
            info = {}
        else:
            # Single environment: convert to scalars for compatibility
            if len(obs.shape) > 1 and obs.shape[0] == 1:
                obs = obs[0]
            if hasattr(reward, '__len__') and len(reward.shape) > 0:
                if reward.shape[0] == 1:
                    reward = reward[0]
            if hasattr(done, '__len__') and len(done.shape) > 0:
                if done.shape[0] == 1:
                    done = done[0]
            
            # Convert to scalar if needed
            if hasattr(reward, 'item'):
                reward = float(reward.item())
            elif hasattr(reward, '__len__') and len(reward) == 1:
                reward = float(reward[0])
            else:
                reward = float(reward)
            
            done = bool(done) if not hasattr(done, '__len__') else bool(done[0] if len(done) > 0 else done)
            truncation = False
            info = {}
        
        return obs, reward, done, truncation, info
    
    @property
    def num_envs(self):
        """Get number of parallel environments."""
        return self.batch_size
    
    @property
    def single_observation_space(self):
        """Get single environment observation space."""
        return self.observation_space
    
    @property
    def single_action_space(self):
        """Get single environment action space."""
        return self.action_space
    
    def close(self):
        """Close the environment (Gymnasium compatibility)."""
        # BRAX environments don't need explicit closing, but we implement for compatibility
        pass


def make_brax_env(env_name: str, num_envs: int = 1, seed: int = 0):
    """
    Create a BRAX environment wrapped for Gymnasium compatibility.
    
    BRAX environments are already vectorized, so we use VectorGymWrapper
    which handles multiple parallel environments efficiently.
    
    Args:
        env_name: BRAX environment name (e.g., "ant", "humanoid", "halfcheetah")
        num_envs: Number of parallel environments (BRAX handles this internally)
        seed: Random seed
    
    Returns:
        Gymnasium-compatible environment
    """
    try:
        from brax import envs
    except ImportError:
        raise ImportError(
            "BRAX is not installed. Install it with: pip install brax jax jaxlib"
        )
    
    # Try different import paths for BRAX gym wrappers (different versions)
    brax_gym = None
    try:
        # Try newer BRAX versions (brax.io.gymnasium)
        from brax.io import gymnasium as brax_gym
    except ImportError:
        try:
            # Try older BRAX versions (brax.io.gym)
            from brax.io import gym as brax_gym
        except ImportError:
            # Fallback: use custom wrapper
            pass
    
    # Create BRAX environment
    # BRAX environments are already vectorized, so num_envs is handled internally
    brax_env = envs.create(env_name=env_name, batch_size=num_envs)
    
    # Wrap BRAX environment to Gymnasium interface
    if brax_gym is not None:
        # Use BRAX's built-in wrapper if available
        try:
            if hasattr(brax_gym, 'VectorGymWrapper'):
                env = brax_gym.VectorGymWrapper(brax_env, seed=seed)
            elif hasattr(brax_gym, 'GymWrapper'):
                env = brax_gym.GymWrapper(brax_env, seed=seed)
            else:
                # Fallback to custom wrapper
                env = BraxToGymnasiumWrapper(brax_env, seed=seed)
        except Exception as e:
            print(f"Warning: BRAX gym wrapper failed ({e}), using custom wrapper")
            env = BraxToGymnasiumWrapper(brax_env, seed=seed)
    else:
        # Use custom wrapper (most reliable)
        print("Using custom BRAX wrapper (brax.io.gym not available)")
        env = BraxToGymnasiumWrapper(brax_env, seed=seed)
    
    return env


def make_brax_env_single(env_name: str, idx: int, capture_video: bool, run_name: str, seed: int = 0):
    """
    Create a single BRAX environment instance (for compatibility with make_env pattern).
    
    BRAX environments are vectorized by default, so this creates a wrapper that can
    be used with SyncVectorEnv, though it's less efficient than using BRAX's native vectorization.
    
    Args:
        env_name: BRAX environment name
        idx: Environment index (for compatibility)
        capture_video: Whether to capture video (not fully supported for BRAX)
        run_name: Run name for logging
        seed: Random seed
    
    Returns:
        Thunk that creates a Gymnasium-compatible environment
    """
    def thunk():
        try:
            from brax import envs
            from brax.io import gym as brax_gym
        except ImportError:
            raise ImportError(
                "BRAX is not installed. Install it with: pip install brax jax jaxlib"
            )
        
        # Create single BRAX environment (batch_size=1)
        brax_env = envs.create(env_name=env_name, batch_size=1)
        
        # Wrap to Gymnasium
        env = brax_gym.GymWrapper(brax_env, seed=seed + idx)
        
        # Add episode statistics wrapper (if available)
        try:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        except Exception:
            pass
        
        return env
    
    return thunk


# List of available BRAX environments
BRAX_ENVIRONMENTS = [
    "ant",           # Ant locomotion task
    "humanoid",       # Humanoid locomotion task
    "halfcheetah",   # HalfCheetah locomotion
    "hopper",        # Hopper locomotion
    "walker2d",       # Walker2D locomotion
    "fetch",         # Fetch manipulation task
    "grasp",         # Grasping task
    "ur5e",          # UR5e robot arm
    "reacher",       # Reacher task
    "acrobot",       # Acrobot swing-up
    "cartpole",      # CartPole (for comparison)
    "inverted_pendulum",  # Inverted Pendulum
    "inverted_double_pendulum",  # Inverted Double Pendulum
]


def get_brax_env_info(env_name: str):
    """
    Get information about a BRAX environment.
    
    Args:
        env_name: BRAX environment name
    
    Returns:
        Dictionary with observation and action space information
    """
    try:
        from brax import envs
    except ImportError:
        raise ImportError("BRAX is not installed")
    
    # Create environment to inspect
    brax_env = envs.create(env_name=env_name, batch_size=1)
    
    # Use custom wrapper to get space information
    env = BraxToGymnasiumWrapper(brax_env, seed=0)
    
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    return {
        "observation_shape": obs_shape,
        "action_shape": action_shape,
        "action_low": action_low,
        "action_high": action_high,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }


if __name__ == "__main__":
    # Example: Test BRAX environment creation
    print("Testing BRAX environment wrapper...")
    
    # Test Ant environment
    print("\n=== Ant Environment ===")
    try:
        env_info = get_brax_env_info("ant")
        print(f"Observation shape: {env_info['observation_shape']}")
        print(f"Action shape: {env_info['action_shape']}")
        print(f"Action range: [{env_info['action_low'][0]:.2f}, {env_info['action_high'][0]:.2f}]")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test Humanoid environment
    print("\n=== Humanoid Environment ===")
    try:
        env_info = get_brax_env_info("humanoid")
        print(f"Observation shape: {env_info['observation_shape']}")
        print(f"Action shape: {env_info['action_shape']}")
        print(f"Action range: [{env_info['action_low'][0]:.2f}, {env_info['action_high'][0]:.2f}]")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nAvailable BRAX environments:")
    for env_name in BRAX_ENVIRONMENTS:
        print(f"  - {env_name}")

