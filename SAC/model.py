# SAC/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy
import os

# --- Constants ---
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# --- Helper Function ---
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, action_low):
        super(Actor, self).__init__()
        
        self.action_dim = action_dim
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # 1. CNN Feature Extractor (same as your DQN)
        self.conv1 = nn.Conv2d(state_dim[0], 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        # 64*8*8 = 4096 features
        
        # 2. MLP Head for Policy
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # 3. Output layers for mean and std
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)
        
        self.apply(weights_init)

    def forward(self, x):
        # CNN
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.reshape(-1, 64*8*8) # Flatten
        
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std

    def sample(self, state, eval=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        
        if eval:
            # For evaluation, we take the mean (deterministic)
            action = mean
        else:
            # For training, we sample (stochastic)
            action = dist.rsample() # reparameterization trick
        
        # Squash action to be between -1 and 1 (using tanh)
        y_t = torch.tanh(action)
        
        # Scale and shift to our action range [action_low, action_high]
        action_scaled = y_t * self.action_scale + self.action_bias
        
        # Calculate log_prob (needed for SAC loss)
        # This formula is the change of variables for log_prob
        log_prob = dist.log_prob(action) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action_scaled, log_prob


# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 Critic
        self.conv1_q1 = nn.Conv2d(state_dim[0], 32, 8, 4)
        self.conv1_q1_bn = nn.BatchNorm2d(32)
        self.conv2_q1 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_q1_bn = nn.BatchNorm2d(64)
        self.conv3_q1 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_q1_bn = nn.BatchNorm2d(64)
        self.fc1_q1 = nn.Linear(64*8*8 + action_dim, 256) # Concatenate action
        self.fc2_q1 = nn.Linear(256, 256)
        self.fc3_q1 = nn.Linear(256, 1)

        # Q2 Critic
        self.conv1_q2 = nn.Conv2d(state_dim[0], 32, 8, 4)
        self.conv1_q2_bn = nn.BatchNorm2d(32)
        self.conv2_q2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_q2_bn = nn.BatchNorm2d(64)
        self.conv3_q2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_q2_bn = nn.BatchNorm2d(64)
        self.fc1_q2 = nn.Linear(64*8*8 + action_dim, 256) # Concatenate action
        self.fc2_q2 = nn.Linear(256, 256)
        self.fc3_q2 = nn.Linear(256, 1)

        self.apply(weights_init)

    def forward(self, state, action):
        # CNN features
        x1 = F.relu(self.conv1_q1_bn(self.conv1_q1(state)))
        x1 = F.relu(self.conv2_q1_bn(self.conv2_q1(x1)))
        x1 = F.relu(self.conv3_q1_bn(self.conv3_q1(x1)))
        x1 = x1.reshape(-1, 64*8*8)
        
        x2 = F.relu(self.conv1_q2_bn(self.conv1_q2(state)))
        x2 = F.relu(self.conv2_q2_bn(self.conv2_q2(x2)))
        x2 = F.relu(self.conv3_q2_bn(self.conv3_q2(x2)))
        x2 = x2.reshape(-1, 64*8*8)

        # Q1 forward
        q1 = torch.cat([x1, action], 1) # Concatenate features and action
        q1 = F.relu(self.fc1_q1(q1))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        # Q2 forward
        q2 = torch.cat([x2, action], 1) # Concatenate features and action
        q2 = F.relu(self.fc1_q2(q2))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2


# --- SAC Agent ---
class SAC(object):
    def __init__(self, state_dim, action_dim, action_high, action_low, params):
        self.device = params['device']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.batch_size = params['batch_size']
        self.policy_freq = params['policy_freq']
        self.autotune_alpha = params['autotune_alpha']
        self.iterations = 0

        self.actor = Actor(state_dim, action_dim, action_high, action_low).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['lr'])
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params['lr'])

        if self.autotune_alpha:
            # Target entropy is heuristic: -|Action|
            self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=params['lr'])
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = params['alpha']
            
    def select_action(self, state, eval=False):
        # Reshape state, convert to tensor, send to device
        state = torch.FloatTensor(state).reshape(1, *state.shape).to(self.device)
        action, _ = self.actor.sample(state, eval=eval)
        # Detach from graph, move to cpu, convert to numpy, flatten
        return action.cpu().detach().numpy().flatten()

    def train(self, replay_buffer):
        self.iterations += 1
        
        # 1. Sample a batch from replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # 2. --- Compute Critic Target (Q-target) ---
        with torch.no_grad():
            # Get next action and log_prob from actor
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # Get Q-values from *target* critic
            q1_target_next, q2_target_next = self.critic_target(next_state, next_action)
            
            # Use the minimum of the two critics (clipped double-Q)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            
            # Add entropy term (the "soft" part)
            q_target_next_soft = q_target_next - self.alpha * next_log_prob
            
            # Compute the final Q-target
            q_target = reward + (1 - done) * self.gamma * q_target_next_soft

        # 3. --- Compute Critic Loss ---
        # Get current Q-values from *main* critic
        q1_current, q2_current = self.critic(state, action)
        
        # MSE loss against the Q-target
        critic_loss_1 = F.mse_loss(q1_current, q_target)
        critic_loss_2 = F.mse_loss(q2_current, q_target)
        critic_loss = critic_loss_1 + critic_loss_2
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. --- Delayed Actor & Alpha Update ---
        if self.iterations % self.policy_freq == 0:
            
            # --- Compute Actor Loss ---
            # Get new action and log_prob from actor
            new_action, new_log_prob = self.actor.sample(state)
            
            # Get Q-values from *main* critic for these new actions
            q1_new, q2_new = self.critic(state, new_action)
            q_new = torch.min(q1_new, q2_new)
            
            # Actor loss is -(Q - alpha * log_prob)
            actor_loss = (self.alpha * new_log_prob - q_new).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- Compute Alpha Loss (if autotuning) ---
            if self.autotune_alpha:
                alpha_loss = -(self.log_alpha * (new_log_prob + self.target_entropy).detach()).mean()
                
                # Optimize alpha
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
            
            # 5. --- Soft Update Target Networks ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        if not os.path.exists('weights'):
            os.makedirs('weights')
            
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
        if self.autotune_alpha:
            torch.save(self.log_alpha, filename + "_log_alpha")
            torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        
        if self.autotune_alpha:
            self.log_alpha = torch.load(filename + "_log_alpha")
            self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))