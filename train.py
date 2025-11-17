import os
from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN
from config import action_map, env_params
# --- MODIFIED IMPORT ---
from utils import save_plots
from environment import SimEnv

def run():
    env = None
    # --- ADD LISTS TO STORE DATA ---
    episode_rewards = []
    epsilon_values = []

    try:
        # --- FIXED BUFFER SIZE ---
        buffer_size = 50000 # (Was 1e6, changed to prevent 122 GiB MemoryError)
        
        batch_size = 32
        state_dim = (128, 128)
        device = "cpu"
        num_actions = len(action_map)
        in_channels = 1
        
        episodes = 5000 # (Example: 5000)

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)

        # --- Create 'graphs' folder ---
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        env = SimEnv(visuals=False, **env_params) # Visuals=False for faster training

        print("Spawning traffic for training...")
        env.create_traffic()
        print("Traffic spawned.")

        for ep in range(episodes):
            print(f"--- Running Training Episode {ep+1}/{episodes} ---")
            
            env.create_player_agent()
            
            # --- CAPTURE THE RETURNED REWARD (eval=False) ---
            ep_reward = env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
            
            # --- STORE DATA FOR PLOTTING ---
            episode_rewards.append(ep_reward)
            epsilon_values.append(model.current_eps) # Get epsilon from the model

            # --- MODIFIED PRINT STATEMENT ---
            avg_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:]) # Avg of last 100
            print(f"--- Ep. {ep+1} FINISHED --- "
                  f"Total Reward: {ep_reward:.2f}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {model.current_eps:.4f} ---")

            env.reset()
        
        # --- SAVE PLOTS AFTER LOOP ---
        print("Training finished. Saving plots...")
        save_plots(range(episodes), episode_rewards, epsilon_values, "training") # <-- CALL save_plots

    except Exception as e:
        print(f"Exception during training: {e}")
    finally:
        # --- Use the new cleanup method ---
        if env is not None:
            try:
                env.cleanup()
            except Exception as e:
                print(f"Exception while cleaning up env: {e}")

if __name__ == "__main__":
    run()