import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import PortfolioAllocationEnv

# --- Configuration ---
DATA_PATH = "oos_predictions.parquet"
MODEL_OUTPUT = "best_ppo_agent.zip"
TOTAL_TIMESTEPS = 250000

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Critical Error: {DATA_PATH} not found. Must complete Step 3 first.")
        sys.exit(1)

    print("Initializing Advanced Financial Management (AFM) PPO Allocator...")
    
    # Instantiate custom execution environment simulating 1 Crore base capital
    env = PortfolioAllocationEnv(
        data_path=DATA_PATH,
        top_k=20,
        rebalance_freq=10, 
        initial_capital=10000000.0,
        tx_cost=0.0015
    )
    
    # Vectorize for Stable-Baselines3 API conformity
    vec_env = DummyVecEnv([lambda: env])
    
    # --- PPO Hyperparameters ---
    # Optimized for noisy financial time-series and strict entropy preservation
    ppo_params = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,            
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,         
        "max_grad_norm": 0.5,
        "verbose": 1
    }
    
    model = PPO(**ppo_params)
    
    print(f"Starting Reinforcement Learning Phase: Executing {TOTAL_TIMESTEPS} market transitions...")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except Exception as e:
        print(f"Training interrupted: {e}")
        sys.exit(1)
        
    print(f"Training Complete. Serializing PyTorch model weights to {MODEL_OUTPUT}...")
    model.save(MODEL_OUTPUT)
    
    # --- Execute Final Validation Walkthrough ---
    obs = vec_env.reset()
    done = False
    
    print("Initiating pure validation matrix evaluation...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)
        done = done[0]
        
    print("RL Agent Compilation Successful.")
