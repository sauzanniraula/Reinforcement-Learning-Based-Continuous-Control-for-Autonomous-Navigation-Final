import os

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN

from config import action_map, env_params
# --- MODIFIED IMPORT ---
from utils import save_plots
from environment import SimEnv

def run():
    env = None
    # --- ADD LIST TO STORE DATA ---
    eval_episode_rewards = []

    try:
        buffer_size = 1e4 # 10,000 for evaluation
        batch_size = 32
        state_dim = (128, 128)
        device = "cpu"
        num_actions = len(action_map)
        in_channels = 1
        
        episodes = 1000 

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)

        saved_model_name = 'weights/model_ep_1000' # <-- CHANGE THIS
        
        try:
            model.load(saved_model_name)
            print(f"Successfully loaded model: {saved_model_name}")
        except FileNotFoundError:
            print(f"Error: Model file not found: {saved_model_name}")
            return # Exit if model not found
            
        # --- Create 'graphs' folder ---
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        env = SimEnv(visuals=True, **env_params)

        print("Spawning traffic for evaluation...")
        env.create_traffic()
        print("Traffic spawned.")

        for ep in range(episodes):
            print(f"--- Running Evaluation Episode {ep+1}/{episodes} ---")
            
            env.create_player_agent()
            
            # --- CAPTURE THE RETURNED REWARD ---
            ep_reward = env.generate_episode(model, replay_buffer, ep, action_map, eval=True)
            
            # --- STORE DATA FOR PLOTTING ---
            eval_episode_rewards.append(ep_reward)

            # --- MODIFIED PRINT STATEMENT ---
            avg_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)
            print(f"--- Eval Ep. {ep+1} FINISHED --- "
                  f"Total Reward: {ep_reward:.2f}, "
                  f"Running Avg Reward: {avg_reward:.2f} ---")

            env.reset()
        
        # --- SAVE PLOTS AFTER LOOP ---
        print("Evaluation finished. Saving plots...")
        save_plots(range(episodes), eval_episode_rewards, None, "evaluation")

    except Exception as e:
        print(f"Exception during evaluation: {e}")
    finally:
        # --- Use the new cleanup method ---
        if env is not None:
            try:
                env.cleanup()
            except Exception as e:
                print(f"Exception while cleaning up env: {e}")

if __name__ == "__main__":
    run()