# SAC/evaluate.py

import os
import sys

# --- Add the project root to the path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------

from SAC.model import SAC
from SAC.replay_buffer import ReplayBuffer # We need a small buffer for env.generate_episode
from config import env_params, sac_params
from utils import save_plots
from environment import SimEnv

def run():
    env = None
    eval_episode_rewards = []

    try:
        # --- Initialize SAC components ---
        # A small buffer is needed, but it won't be used for training
        replay_buffer = ReplayBuffer(
            sac_params['state_dim'], 
            sac_params['action_dim'], 
            sac_params['batch_size'], 
            1000, # Small buffer size for eval
            sac_params['device']
        )
        
        model = SAC(
            sac_params['state_dim'],
            sac_params['action_dim'],
            sac_params['action_high'],
            sac_params['action_low'],
            sac_params
        )
        
        # --- Load Your Trained Model ---
        # Change this to the model you saved from training
        saved_model_name = 'weights/sac_model_ep_1000' # <-- CHANGE THIS
        
        try:
            model.load(saved_model_name)
            print(f"Successfully loaded model: {saved_model_name}")
        except FileNotFoundError:
            print(f"Error: Model file not found: {saved_model_name}")
            return
            
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # --- Initialize Environment ---
        # Set visuals=True to watch your agent drive!
        env = SimEnv(visuals=True, **env_params)

        print("Spawning traffic for evaluation...")
        env.create_traffic()
        print("Traffic spawned.")
        
        episodes = 100 # Run 100 test episodes

        for ep in range(episodes):
            print(f"--- Running Evaluation Episode {ep+1}/{episodes} ---")
            
            env.create_player_agent()
            
            # --- Run Episode ---
            # We pass eval=True to use deterministic (mean) actions
            ep_reward = env.generate_episode(model, replay_buffer, ep, action_map=None, eval=True)
            
            eval_episode_rewards.append(ep_reward)

            avg_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)
            print(f"--- Eval Ep. {ep+1} FINISHED --- "
                  f"Total Reward: {ep_reward:.2f}, "
                  f"Running Avg Reward: {avg_reward:.2f} ---")

            env.reset()
        
        print("Evaluation finished. Saving plots...")
        save_plots(range(episodes), eval_episode_rewards, None, "sac_evaluation")

    except Exception as e:
        print(f"Exception during evaluation: {e}")
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception as e:
                print(f"Exception while cleaning up env: {e}")

if __name__ == "__main__":
    run()