# SAC/train.py

import os
import sys

# --- Add the project root to the path ---
# This allows us to import 'environment' and 'utils'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------

from SAC.model import SAC
from SAC.replay_buffer import ReplayBuffer
from config import env_params, sac_params
from utils import save_plots # We can still use our plotting util
from environment import SimEnv

def run():
    env = None
    episode_rewards = []
    
    try:
        # --- Initialize SAC components ---
        replay_buffer = ReplayBuffer(
            sac_params['state_dim'], 
            sac_params['action_dim'], 
            sac_params['batch_size'], 
            sac_params['buffer_size'], 
            sac_params['device']
        )
        
        model = SAC(
            sac_params['state_dim'],
            sac_params['action_dim'],
            sac_params['action_high'],
            sac_params['action_low'],
            sac_params
        )
        
        # --- Create 'graphs' folder ---
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # --- Initialize Environment ---
        env = SimEnv(visuals=False, **env_params) # Visuals=False for faster training

        print("Spawning traffic for training...")
        env.create_traffic()
        print("Traffic spawned.")

        episodes = 5000 # No. of episodes you want to use 

        for ep in range(episodes):
            print(f"--- Running Training Episode {ep+1}/{episodes} ---")
            
            env.create_player_agent()
            
            # --- Run Episode ---
            # We pass eval=False to use stochastic actions
            ep_reward = env.generate_episode(model, replay_buffer, ep, action_map=None, eval=False)
            
            episode_rewards.append(ep_reward)

            # --- Print Episode Summary ---
            avg_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
            print(f"--- Ep. {ep+1} FINISHED --- "
                  f"Total Reward: {ep_reward:.2f}, "
                  f"Avg Reward (last 100): {avg_reward:.2f} ---")

            env.reset()
        
        # --- Save Plots After Loop ---
        print("Training finished. Saving plots...")
        # We pass 'None' for epsilon, as SAC doesn't use it
        save_plots(range(episodes), episode_rewards, None, "sac_training")

    except Exception as e:
        print(f"Exception during training: {e}")
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception as e:
                print(f"Exception while cleaning up env: {e}")

if __name__ == "__main__":
    run()
