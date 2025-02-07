# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# https://github.com/chingyaoc/pytorch-REINFORCE/tree/master
# https://blog.naver.com/songblue61/221853600720
# While input training, x.shape: ([1, 300, 4]), hidden[0].shape: ([1, 1, 32]), hidden[1].shape: ([1, 1, 32])

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Bernoulli
from collections import deque
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from mfrl_lib.lib import *

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class Pinet(nn.Module):
    def __init__(self, n_observations, n_nodes):
        super(Pinet, self).__init__()
        self.hidden_space = 32
        # Input: state (3) + age (1) for each device + counter
        self.lstm = nn.LSTM(n_observations, self.hidden_space, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_space, self.hidden_space)
        # Output separate probabilities for each device's action
        self.actor = nn.Linear(self.hidden_space, n_nodes)
        self.critic = nn.Linear(self.hidden_space, 1)
        
        self.init_weights()
        
    def init_weights(self):
        for layer in [self.fc1, self.actor, self.critic]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.)

    def pi(self, x, hidden):
        x, lstm_hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x))
        x = self.actor(x)
        prob = F.softmax(x, dim=2)
        prob = F.sigmoid(x)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x, lstm_hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x))
        v = self.critic(x)
        return v

    def sample_action(self, obs, h, c):
        prob, (h_new, c_new) = self.pi(obs, (h, c))
        # Sample binary action for each node
        dist = Bernoulli(prob)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(-1)  # Sum log probs across nodes
        entropy = dist.entropy().mean()
        value = self.v(obs, (h, c))
        return prob, actions, h_new, c_new, log_prob, entropy, value
    
class Agent:
    def __init__(self, topology, n_obs, n_act, arrival_rate):
        self.gamma = GAMMA
        self.topology = topology
        self.arrival_rate = arrival_rate
        self.n = topology.n
        self.env = MFRLFullEnv(self)
        self.pinet = Pinet(n_obs, n_act).to(device)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=LEARNING_RATE)
        self.data = []

    def flatten_observation(self, obs):
        # Flatten the observation dictionary into a single vector
        flat_obs = []
        for device_obs in obs["devices"]:
            flat_obs.extend(device_obs["state"])  # 3 values
            flat_obs.append(device_obs["age"][0])  # 1 value
        flat_obs.append(obs["counter"])  # 1 value
        return np.array(flat_obs)
        
    def put_data(self, item):
        self.data.append(item)

    def train(self):
        s, a, r, s_prime, done_mask = self.make_batch()
        h = (torch.zeros([1, 1, self.pinet.hidden_space], dtype=torch.float).to(device),
            torch.zeros([1, 1, self.pinet.hidden_space], dtype=torch.float).to(device))
        
        # Calculate advantage as before
        v_prime = self.pinet.v(s_prime, h).squeeze(1).detach()
        td_target = r + self.gamma * v_prime * done_mask
        v_s = self.pinet.v(s, h).squeeze(1)
        delta = td_target - v_s
        advantage = delta.detach()

        # Get action probabilities
        pi, _ = self.pinet.pi(s, h)
        pi = pi.squeeze(0)  # Remove batch dimension if batch_size=1
        
        # # Make binary actions to decimal
        # a = binary_array_to_decimal(a).unsqueeze(1).to(device)
        
        # pi_a = pi.gather(1, a)
        # loss = -torch.log(pi_a) * advantage.unsqueeze(1) + F.smooth_l1_loss(v_s, td_target.detach())
        
        action_loss = F.binary_cross_entropy_with_logits(
            pi, a.float(), reduction='none').mean(-1)  # Average across nodes
        
        loss = action_loss * advantage + F.smooth_l1_loss(v_s, td_target.detach())
        

        # Optimize
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.pinet.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.data = []

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s = torch.tensor(s_lst, dtype=torch.float).to(device).view([1, len(self.data), -1])
        a = torch.stack(a_lst).to(device).squeeze()  # Stack and remove middle dimension
        r = torch.tensor(r_lst).to(device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(device).view([1, len(self.data), -1])
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(device)
        return s, a, r, s_prime, done_mask


if __name__ == "__main__":
    # Summarywriter setting
    timestamp = FIXED_TIMESTAMP
    output_path = 'outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = SummaryWriter(output_path + "/" + "RA2Cfull" + "_" + topo_string + "_" + timestamp)
    
    n_obs = 4*node_n+1
    # n_act = 2**node_n
    n_act = node_n
    
    # Make agent
    agent = Agent(topology, n_obs=n_obs, n_act=n_act, arrival_rate=arrival_rate)

    reward_data = []
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    best_model_state = None
    best_reward = -float('inf')

    for n_epi in tqdm(range(MAX_EPISODES), desc="Episodes", position=0, leave=True):
        episode_utility = 0.0
        observation, _ = agent.env.reset(seed=GLOBAL_SEED)
        flat_obs = agent.flatten_observation(observation)
        h = torch.zeros(1, 1, 32).to(device)
        c = torch.zeros(1, 1, 32).to(device)
        done = False
        
        for t in tqdm(range(MAX_STEPS), desc="Steps", position=1, leave=False):
            obs_tensor = torch.from_numpy(flat_obs.astype('float32')).unsqueeze(0).unsqueeze(0).to(device)
            prob, actions, h, c, log_prob, entropy, value = agent.pinet.sample_action(obs_tensor, h, c)
            
            # Get binary actions for each device
            # actions = decimal_to_binary_array(a.item(), node_n)
            # actions = np.array(actions.cpu().int())
            next_observation, reward, done, _, _ = agent.env.step(actions)
            next_flat_obs = agent.flatten_observation(next_observation)
            
            episode_utility += reward
            reward_data.append({
                'episode': n_epi,
                'step': t,
                'reward': reward/node_n,
                'action': actions,
                'age': agent.env.age.copy()
            })
            
            agent.put_data((flat_obs, actions, reward, next_flat_obs, done))
            flat_obs = next_flat_obs
                
            if done:
                break
            
        agent.train()
        episode_utility /= node_n
        writer.add_scalar('Avg. Rewards per episodes', episode_utility, n_epi)

        if n_epi % print_interval == 0:
            print(f"# of episode :{n_epi}, avg reward : {episode_utility:.1f}")
            
        if episode_utility > best_reward:
            best_reward = episode_utility
            best_model_state = agent.pinet.state_dict()
            
        if early_stopping.should_stop(episode_utility):
            print(f"Early stopping at episode {n_epi}")
            agent.pinet.load_state_dict(best_model_state)
            break

    # Save rewards to DataFrame and CSV
    csv_path = 'csv'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    csv_path = f'{csv_path}/RA2CFull_{topo_string}_{timestamp}.csv'
    
    reward_df = pd.DataFrame(reward_data)
    # Convert actions to numpy arrays and reshape
    reward_df['action'] = reward_df['action'].apply(lambda x: x.cpu().numpy().flatten())
    # Process columns
    action_cols = pd.DataFrame(reward_df['action'].tolist(), columns=[f'action_{i}' for i in range(agent.n)])
    age_cols = pd.DataFrame(reward_df['age'].tolist(), columns=[f'age_{i}' for i in range(agent.n)])
    # Combine into final dataframe
    result_df = pd.concat([reward_df.drop(['action', 'age'], axis=1), action_cols, age_cols], axis=1)
    result_df.to_csv(csv_path, index=False)
    writer.close()

    # Save models
    model_path = 'models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_model(agent.pinet, f'{model_path}/RA2CFull_{topo_string}_{timestamp}.pth')