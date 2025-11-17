# config.py

import torch  # <-- THIS IS THE FIX

# --- Environment Parameters ---
env_params = {
    'target_speed' : 30, 
    'max_iter': 4000,
    'start_buffer': 10,       
    
    # --- HERE ARE THE NEW SPECS FOR SMOOTHNESS ---
    'train_freq': 100,        # Train only ONCE every 100 steps (was 10)
    'train_batch_count': 10,  # But when we train, do it 10 times in a row
    # --- END OF FIX ---
    
    'save_freq': 200,         
    'start_ep': 0,
    'max_dist_from_waypoint': 20
}
# --- SAC Algorithm Parameters ---
sac_params = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'lr': 3e-4,                 # Learning rate for all networks
    'gamma': 0.99,              # Discount factor
    'tau': 0.005,               # Target network soft update rate
    'batch_size': 32,
    'buffer_size': 50000,       # Reduced from 1e6 to prevent memory errors
    
    # State/Action Dimensions
    'state_dim': (1, 128, 128), # (Channels, Height, Width)
    'action_dim': 1,            # 1 continuous action (steering)
    'action_high': 0.75,        # Max steering value
    'action_low': -0.75,        # Min steering value
    
    # SAC Specific
    'policy_freq': 2,           # Frequency to update actor (every 2 critic updates)
    'alpha': 0.2,               # Initial entropy temperature
    'autotune_alpha': True      # Automatically tune the alpha parameter
}