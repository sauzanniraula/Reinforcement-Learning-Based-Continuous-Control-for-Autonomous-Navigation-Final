Self-Driving Agent in CARLA (DQN and SAC)
This project trains a deep reinforcement learning agent to drive autonomously in the CARLA simulator. It has been updated from a simple DQN to a state-of-the-art SAC algorithm for smoother, more effective, continuous control.

This repository contains two different agent algorithms:

DQN (Deep Q-Network): The original algorithm. A good starting point, but limited to discrete steering actions (e.g., "turn left 0.5" or "go straight").

SAC (Soft Actor-Critic): (Recommended) A modern, state-of-the-art algorithm that learns continuous steering (e.g., "turn 0.182"). This results in much smoother, more human-like driving.

Project Structure
self_driving_agent/
├── DQN_Control/           # Logic for the (Old) DQN Algorithm
│   ├── model.py
│   └── replay_buffer.py
├── SAC/                   # Logic for the (New) SAC Algorithm
│   ├── model.py           # Contains the Actor & Critic network logic
│   ├── replay_buffer.py   # Buffer modified for continuous actions
│   ├── train.py           # <-- Run this to TRAIN the SAC agent
│   └── evaluate.py        # <-- Run this to TEST the SAC agent
├── weights/               # Saved models for both algorithms
├── graphs/                # Saved plots for both algorithms
├── config.py              # Main configuration file (for Env, SAC, etc.)
├── environment.py         # CARLA simulation environment (used by both)
├── train.py               # <-- Run this to TRAIN the (Old) DQN agent
├── DQN_main.py            # <-- Run this to TEST the (Old) DQN agent
├── utils.py               # Helper functions (plotting, image processing)
├── initial_setup.py       # Run once to create the 'weights' folder
└── README.md              # This file
1. Setup & Installation
Prerequisites:

This project is built for CARLA 0.9.16.

You must have an NVIDIA GPU for the SAC algorithm to train effectively.

Installation Steps:

Create weights Folder: Run the setup script one time to create the weights folder.

Bash

python initial_setup.py
Install Dependencies: Activate your virtual environment (pme_venv) and install the required packages.

Bash

# Make sure you are in your virtual environment (e.g., pme_venv)
pip install pygame torch opencv-python numpy matplotlib
2. How to Run: The 3-Step Workflow
A Python script acts as a "client" that connects to the CARLA "server." You must always start the CARLA simulator first.

Step 1: Start the CARLA Server (Terminal 1)
Open a new PowerShell terminal and launch the simulator.

For Training (Fastest): Run CARLA in "headless" mode (no window). This is highly recommended for training.

PowerShell

# Navigate to your CARLA folder
cd "C:\Carla\CARLA_0.9.16"

# Run headless with OpenGL (Click OK on the warning)
.\CarlaUE4.exe -opengl
For Testing/Watching (Slower): Run CARLA in a small window.

PowerShell

# Navigate to your CARLA folder
cd "C:\Carla\CARLA_0.9.16"

# Run in a window
.\CarlaUE4.exe -windowed -ResX=800 -ResY=600
Leave this terminal running in the background.

Step 2: Start Your Python Script (Terminal 2)
Open a second PowerShell terminal.

Navigate to your project folder:

PowerShell

cd C:\projects\self_driving_agent
Activate your virtual environment:

PowerShell

.\pme_venv\Scripts\Activate
Step 3: Choose Your Action
To Train: This is the "school" where your agent learns. This will take many hours.

Bash

# To train the (Recommended) SAC Agent:
python SAC/train.py

# To train the (Old) DQN Agent:
python train.py
To Evaluate (The "Driving Test"): This loads a saved "brain" from the weights/ folder and lets you watch it drive.

CRITICAL: Before running, you must open the evaluation script (SAC/evaluate.py or DQN_main.py) and change the saved_model_name variable to match the model you just trained.

Bash

# To test the SAC Agent (with visuals):
python SAC/evaluate.py

# To test the DQN Agent (with visuals):
python DQN_main.py
3. 🚨 CRITICAL TROUBLESHOOTING
Problem: You try to run a script and it immediately crashes with RuntimeError: time-out of 10000ms... or connection failed...

Cause: A "stuck" or "zombie" CARLA process from a previous run is blocking port 2000.

Solution (The "Golden Rule"):

Find the "Stuck" Process: Run this in your terminal to find all processes using port 2000.

PowerShell

netstat -ano | findstr :2000
You will see an output like this. The last number is the Process ID (PID). TCP 0.0.0.0:2000 0.0.0.0:0 LISTENING 13600 (In this example, the PID is 13600)

Kill the Process: Use the PID you just found to force-kill the process.

PowerShell

taskkill /F /PID 13600
(Replace 13600 with the number you found)

After you see SUCCESS, the port is free. You can now go back to Step 1 and start the CARLA server again.

4. Understanding the Output
Terminal Log
During training, you will see output that shows the agent's progress.

Bash

--- Running Training Episode 18/5000 ---
  [Ep: 18, Step: 100/4000] Running Ep. Reward: -6.22
  [Ep: 18, Step: 161] --- STARTING TRAINING ---
  [Ep: 18, Step: 161] --- TRAINING FINISHED ---
Episode 17 processed 172
--- Ep. 18 FINISHED --- Total Reward: -238.00, Avg Reward (last 100): -190.14 ---
Episode: One full "life" or driving attempt.

Step: A single decision or frame within that episode.

--- STARTING TRAINING ---: This is the "lag." The simulation pauses to train the AI. This is controlled by train_freq in config.py.

Saved Files
Models (weights/ folder):

The train.py scripts save your model's "brain" here every save_freq episodes.

DQN: model_ep_1000_Q, model_ep_1000_optimizer

SAC: sac_model_ep_1000_actor, sac_model_ep_1000_critic

Graphs (graphs/ folder):

At the end of a run, scripts save .png plots.

sac_training_rewards.png: The most important graph. It shows the total reward per episode. You want to see this line go up over time.

sac_evaluation_rewards.png: A bar chart showing the final scores for your "test drives." This is your model's "report card."

5. (Advanced) Configuration
All important settings are in config.py.

target_speed: Speed (km/h) you want the car to try and maintai.

max_iter: Maximum number of steps in one episode before it resets.

start_buffer: Number of episodes to run before training begins.

train_freq: How often to train (e.g., train every 100 steps).

train_batch_count: How many times to train when train_freq is hit.

save_freq: How often to save a checkpoint model (e.g., every 200 episodes).
