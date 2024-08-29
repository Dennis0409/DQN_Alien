from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time
import os

from memory import ReplayMemory
from models import NoisyDQN,NoisyLinear, DQN # Adjust based on your actual model definitions
from game_small_maze import Maze

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import data
# import levenshtein_distance
# import optics


Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

# Constants
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 370 
CELL_SIZE = 20
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT - 50) // CELL_SIZE  # Adjusted height for timer display
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (192, 192, 192)
BLUE = (0, 0, 255)

reward_list = []
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    
    return action

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device).view(-1, 1)  # Ensure shape [BATCH_SIZE, 1]
    reward_batch = torch.cat(batch.reward).to(device)

    # print("state_batch shape:", state_batch.shape)
    # print("action_batch shape:", action_batch.shape)

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_action_values = policy_net(state_batch)
    # print("state_action_values shape:", state_action_values.shape)
    
    state_action_values = state_action_values.gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()

def get_state(obs, episode, save_image=False):
    map_info = obs['maze'] 
    #-->地圖資訊
    map_array= np.array(map_info)
    normalized_map = map_array / np.max(map_array)
    
    state = torch.tensor(normalized_map, dtype=torch.float).unsqueeze(0) 
    state = state.unsqueeze(0) 

    if save_image:
        plt.imshow(normalized_map, cmap='gray')
        filename = f'./temp/{episode}/{steps_done}.png'
        dir = os.path.dirname(filename)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(filename)
        plt.close()
    
    return state


def train(env, n_episodes, render=False):
    all_losses = []
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs, episode, False)
        total_reward = 0.0
        episode_loss = 0.0
             
        for t in count():
            action = select_action(state)
            if render:
                 env.render()
            obs, reward, done = env.step(action.item())
            #print(f"Episode {episode}, Step {t}, Reward: {reward}")
            total_reward += reward

            if not done:
                next_state = get_state(obs, episode, False)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cuda'), next_state, reward.to('cuda'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                loss = optimize_model()
                if loss is not None:
                    episode_loss += loss

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        
        avg_loss = episode_loss / (t + 1) if t > 0 else 0
        all_losses.append(avg_loss)
        reward_list.append(total_reward)
        print(f'Episode {episode}/{n_episodes} \t Total reward: {total_reward} \t Average loss: {avg_loss:.4f}')
    
    env.close()
    
    # Plotting the loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_episodes), all_losses, label='Average Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_per_episode_S5001.png')  # Save plot as an image file


    # Assuming you have a list `all_rewards` that tracks the rewards per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_episodes),reward_list, label='Reward')  # Use all_rewards instead of all_losses
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_per_episode_S5001.png')  # Save plot as an image file

if __name__ == '__main__':
    tt = False
    # if tt:
    #     optics.optic().clustering_from_mem()
    #     exit()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    
    # rendering_enabled 選擇要不要顯示
    env = Maze(countdown_time=30,rendering_enabled=False)
    
    # Define the number of actions in your environment
    n_actions = 4  # Adjust this based on your environment's action space

    # Create networks
    policy_net = DQN(n_actions=n_actions).to(device)
    target_net = DQN(n_actions=n_actions).to(device)
    # policy_net = NoisyDQN(n_actions=n_actions).to(device)
    # target_net = NoisyDQN(n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    steps_done = 0

    # Initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Train model
    train(env, 5001)
    torch.save(policy_net.state_dict(), "DQN_Smaze_5001")
    
  
    # dis_graph = levenshtein_distance.PathDistanceCalculator().calculate_distances(path_array)
    # print(dis_graph)
    # print("----------------------------------------------")
    

    # # Save distance graph
    # with open('test.mem', 'w') as f:
    #     for i in dis_graph:
    #         for j in i:
    #             f.write(str(int(j)) + " ")
    #         f.write("\n")
    
    # optics.optic().clustering(dis_graph)
