# rl_tutor/agent.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import os
from . import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, n_latent_var)
        self.layer2 = nn.Linear(n_latent_var, n_latent_var)
        self.action_layer = nn.Linear(n_latent_var, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.tanh(self.layer1(state))
        x = self.tanh(self.layer2(x))
        action_probs = torch.softmax(self.action_layer(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim, n_latent_var):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, n_latent_var)
        self.layer2 = nn.Linear(n_latent_var, n_latent_var)
        self.value_layer = nn.Linear(n_latent_var, 1)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.tanh(self.layer1(state))
        x = self.tanh(self.layer2(x))
        state_value = self.value_layer(x)
        return state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim, config.NETWORK_SIZE).to(device)
        self.critic = Critic(state_dim, config.NETWORK_SIZE).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LEARNING_RATE)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=config.CRITIC_LEARNING_RATE)
        
        self.actor_old = Actor(state_dim, action_dim, config.NETWORK_SIZE).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def act(self, state, valid_actions):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_probs = self.actor_old(state)
        
        mask = torch.full_like(action_probs, float('-inf'))
        mask[:, valid_actions] = 0
        masked_action_probs = action_probs + mask
        
        final_probs = torch.softmax(masked_action_probs, dim=-1)

        dist = Categorical(final_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.detach()

    def learn(self, states, actions, rewards, next_states, dones, log_probs):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(np.array(dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(device)
        
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            
            deltas = rewards + config.GAMMA * next_values * (~dones) - values
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            for t in reversed(range(len(rewards))):
                advantages[t] = deltas[t] + config.GAMMA * config.GAE_LAMBDA * last_advantage * (~dones[t])
                last_advantage = advantages[t]
            
            returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(config.EPOCHS_PER_UPDATE):
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            
            log_probs_new = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(states).squeeze()
            
            ratios = torch.exp(log_probs_new - old_log_probs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - config.CLIP_EPSILON, 1 + config.CLIP_EPSILON) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, returns)
            entropy_loss = -config.ENTROPY_BETA * dist_entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
        self.actor_old.load_state_dict(self.actor.state_dict())

    def save(self):
        actor_path = os.path.join(config.MODEL_DIR, "ppo_actor.pth")
        critic_path = os.path.join(config.MODEL_DIR, "ppo_critic.pth")
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Models saved to {config.MODEL_DIR}")

    def load(self):
        actor_path = os.path.join(config.MODEL_DIR, "ppo_actor.pth")
        critic_path = os.path.join(config.MODEL_DIR, "ppo_critic.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))
        self.actor_old.load_state_dict(self.actor.state_dict())

