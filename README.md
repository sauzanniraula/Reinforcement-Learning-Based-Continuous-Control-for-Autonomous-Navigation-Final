
Self-Driving Agent in CARLA (DQN and SAC)

This project trains a deep reinforcement learning agent to drive autonomously in the CARLA simulator. It has been updated to be highly stable, crash-resistant, and provides detailed performance graphs.

This repository contains two different agent algorithms:

DQN (Deep Q-Network): The original algorithm. This is a good starting point but is limited to discrete steering actions (e.g., "turn left 0.5" or "go straight").

SAC (Soft Actor-Critic): (Recommended) A modern, state-of-the-art algorithm that learns continuous steering (e.g., "turn 0.182"). This results in much smoother, more human-like driving.

Project Structure

self_driving_agent/
├── DQN_Control/           # Logic for the DQN Algorithm
│   ├── model.py
│   └── replay_buffer.py
├── SAC/                   # Logic for the new SAC Algorithm
│   ├── model.py
│   ├── replay_buffer.py
│   ├── train.py           # <-- Run this to TRAIN the SAC agent
│   └── evaluate.py        # <-- Run this to TEST the SAC agent
├── weights/               # Saved models for both algorithms
├── graphs/                # Saved plots for both algorithms
├── config.py              # Main configuration file
├── environment.py         # CARLA simulation environment (used by both)
├── train.py               # <-- Run this to TRAIN the DQN agent
├── DQN_main.py            # <-- Run this to TEST the DQN agent
├── utils.py               # Helper functions (plotting, image processing)
└── README.md              # This file


1. Setup & Installation

Install CARLA: This project is built for CARLA 0.9.16.

Install Dependencies: Open your PowerShell, activate your environment, and install the required packages.

# Make sure you are in your virtual environment (e.g., pme_venv)
pip install pygame torch opencv-python numpy matplotlib


Create Folders: Run the setup script one time to create the weights folder.

python.exe initial_setup.py


2. How to Run: The 3-Step Workflow

A Python script acts as a "client" that connects to the CARLA simulator "server." You must always start the CARLA simulator first.

Step 1: Start the CARLA Server (Terminal 1)

Open a new PowerShell terminal.

Run the executable. This command will open it in a windowed, 800x600 mode.

& "C:\Carla\CARLA_0.9.16\CarlaUE4.exe" -windowed -ResX=800 -ResY=600


Wait for the simulator window to open and fully load the Town02_Opt map.

Step 2: Train Your Agent (Terminal 2)

This is the "School" where your agent learns to drive. This will take many hours to produce a good model.

Open a second PowerShell terminal.

Navigate to your project folder:

cd C:\projects\self_driving_agent


Activate your virtual environment:

.\pme_venv\Scripts\Activate


Run the training script. (Running the SAC script is recommended).

To train the new (Recommended) SAC Agent:

python SAC/train.py


To train the old DQN Agent:

python train.py


Step 3: Evaluate Your Agent (The "Driving Test")

This is the "implementation" of your final model. It loads the "brain" you saved from training and runs a few "test drives" to see how well it performs.

CRITICAL: Before running, you must open the evaluation script (SAC/evaluate.py or DQN_main.py) and change the saved_model_name variable to match the model you just trained.

For example, if train.py saved model_ep_1000, you must change DQN_main.py to:

saved_model_name = 'weights/model_ep_1000' # <-- CHANGE THIS


Make sure your CARLA Server (from Step 1) is still running.

In your (pme_venv) terminal, run the evaluation script:

To test the SAC Agent (with visuals):

python SAC/evaluate.py


To test the DQN Agent (with visuals):

python DQN_main.py


A Pygame window will open, and you can watch your trained Tesla drive!

3. CRITICAL TROUBLESHOOTING

This is the fix for the most common error you faced.

Problem: You try to run train.py or DQN_main.py and it immediately crashes with:

RuntimeError: time-out of 10000ms while waiting for the simulator...

connection failed: No connection could be made...

Cause: A "stuck" or "zombie" CARLA process from a previous run is blocking port 2000, and the new script can't connect.

How to Fix (The "Golden Rule"):
Before starting a new run, close the CARLA window and run these two commands in your (pme_venv) terminal to clear the port.

Find the "stuck" process:

netstat -ano | findstr :2000


Look at the output. You will see a line with LISTENING and a 5-digit number at the very end. This is the Process ID (PID).

TCP 0.0.0.0:2000 0.0.0.0:0 LISTENING 13600 (In this example, the PID is 13600)

Kill the process:

Use the PID you just found.

taskkill /F /PID 13600


(Replace 13600 with the number you found).

After you see SUCCESS, the port is clear. You can now go back to Step 1 and start the server again.

4. Understanding the Output

Terminal Log

During Training (train.py or SAC/train.py): The terminal will show your progress and checkpoint saves.

Spawning 30 NPC vehicles...
Successfully spawned 30 NPC vehicles.
Traffic spawned.
Completed 10/1000 episodes. Last reward: -50.21
Completed 20/1000 episodes. Last reward: -112.45
...
--- Checkpoint saved at episode 200 ---
...
Training finished. Saving plots...
Saving final model (model_ep_1000)...
Destroying all actors...


During Evaluation (DQN_main.py or SAC/evaluate.py): The terminal will show the results of your "test drives."

Successfully loaded model: weights/model_ep_1000
Spawning traffic for evaluation...
...
Traffic spawned.
--- Running Evaluation Episode 1/5 ---
Episode 1 Finished. Total Reward: 30.88
--- Running Evaluation Episode 2/5 ---
Episode 2 Finished. Total Reward: 18.33
...
Evaluation finished. Saving plots...
Destroying all actors...


Saved Files (Your "Final Model")

This answers your most important question: "where is the final model and how do I implement it?"

Saved Models (in weights/ folder):

Saving: The train.py or SAC/train.py script saves the one, final, "smart" brain at the very end of all episodes.

Files:

DQN: model_ep_1000_Q

SAC: sac_model_ep_1000_actor

Implementation: The DQN_main.py and SAC/evaluate.py scripts are the implementation. You use them to load this "final brain" and watch it drive.

Saved Graphs (in graphs/ folder):

training_rewards_plot.png: The most important graph. It shows the reward for each episode. You want to see the red "moving average" line go up over time. This proves it is learning.

training_epsilon_plot.png: (DQN Only) Shows the "learning rate" (exploration) dropping from 1.0 (random) to 0.05 (smart).

evaluation_rewards_plot.png: A bar chart showing the final score for your 5 "test drives." This is your model's "report card."

# How to Run

## Setup
I did not create a dependency or yml file (will do so at a later time), but you need carla, pygame, pytorch, opencv and numpy to run this project

You should ensure that you have a `weights` folder when you run the project. If you do not have one, then just run `initial_setup.py` and it will create it for you. If you just cloned the repository, I reccomend you run this file first.

## main.py
Run this file if you want to evaluate the performance of your agent
```
env = SimEnv(visuals=False)
```

The call above initializes our simulation environment. You should set visuals to `False` if you do not want to open this with pygame, or to `True` if you want a pygame window to open along with the simulator.

```
model.load('weights/model_ep_4400')
```

This loads a trained/pre-trained model. The program will not run unless it can load this model.
The 4400 indicates that this model was trained for 4400 episodes.
For example, if you train your own model for 200 episodes you will see the following files in the weights folder

`model_ep_200_optimizer` and `model_ep_200_Q`

You can then load the model as `model.load('weights/model_ep_200')`. Please note however that this is likely to be a very bad model, and it will learn effectively after many episodes.

## train.py
This is for training the model. The model only starts learning after a certain number of episodes, and it can take from 8-10 hours (at least on my setup) before we see signs of learning. I will now describe a few variables you can set to configure your training process. You can modify them yourself in `config.py`.

`target_speed` --> Speed you want the car to move at in km/h

`max_iter` --> Maximum number of steps before starting a new episode

`start_buffer` --> Number of episodes to run before starting training

`train_freq` --> How often to train (set to 1 to train every step, 2 to train every 2 steps etc)

`save_freq`: --> Frequency of saving our model

`start_ep` --> Which episode should we start on (just a counter which you can update if program crushes while training for example)

`max_dist_from_waypoint` --> Maximum distance from waypoint/road before we decide to terminate the episode









## Some Command: 

Absolutely. That's a great idea. It's much easier to have a simple command list.

Here are all the commands you will need for this project, including the troubleshooting ones for when the server crashes.

One-Time Setup (Run This First)
Activate Your Environment (if not already active):

* PowerShell

.\pme_venv\Scripts\Activate
Install All Dependencies:

* PowerShell

pip install pygame torch opencv-python numpy matplotlib
Create the weights Folder:

* PowerShell

python.exe initial_setup.py
The 3-Step Workflow (Run This Every Time)
This is the standard process for training or testing.

Step 1: Start CARLA Server (In Terminal 1)

Start the simulator and wait for the map to load.

* PowerShell

& "C:\Carla\CARLA_0.9.16\CarlaUE4.exe" -windowed -ResX=800 -ResY=600
Step 2: Activate & Run Python (In Terminal 2)

Navigate to your project folder and activate your environment.

* PowerShell

cd C:\projects\self_driving_agent
.\pme_venv\Scripts\Activate
Step 3: Choose Your Script

To Train the (Recommended) SAC Agent:

* PowerShell

python SAC/train.py
To Train the (Old) DQN Agent:

* PowerShell

python train.py
To Test/Evaluate the SAC Agent: (Remember to edit SAC/evaluate.py to load the correct model first!)

* PowerShell

python SAC/evaluate.py
To Test/Evaluate the DQN Agent: (Remember to edit DQN_main.py to load the correct model first!)

* PowerShell

python DQN_main.py
CRITICAL TROUBLESHOOTING (How to Fix Crashes)
You mentioned this is a problem. Use this "Golden Rule" workflow every time the script crashes or you get a RuntimeError: time-out or connection refused error.

Step 1: Find the "Stuck" Process

Run this to find all processes using port 2000. Look at the last number on the line.

*PowerShell

netstat -ano | findstr :2000
You will see something like: TCP 0.0.0.0:2000 0.0.0.0:0 LISTENING 13600

The Process ID (PID) is 13600.

Step 2: Kill the "Stuck" Process

Use the PID you just found.

* PowerShell

taskkill /F /PID 13600
(Replace 13600 with whatever number you found)

After you see SUCCESS, the port is free. You can now go back and start the CARLA server again.