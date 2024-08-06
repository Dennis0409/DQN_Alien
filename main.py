import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time


import gym
from wrappers import *

import matplotlib.pyplot as plt
import gym
from wrappers import *
import os
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


import data
import levenshtein_distance
import optics

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

reward_list = []
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    
    
    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

#
def get_state(obs,ep,p):
    #print(obs)
    state = np.array(obs)
    if(p==True):
        plt.imshow(state)
        filename=f'./temp/{ep}/{steps_done}.png'
        dir=os.path.dirname(filename)
        if(dir and not os.path.exists(dir)):
            os.makedirs(dir)

        plt.savefig(filename)
        plt.close()
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    
    return state.unsqueeze(0)

def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        obs= env.reset()
        
        state = get_state(obs[0],episode,False)
        total_reward = 0.0
        for t in count():
            
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info ,_= env.step(action)
            
            total_reward += reward

            if not done:
                next_state = get_state(obs,episode,False)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cuda'), next_state, reward.to('cuda'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        ##print(f'train end ep')
        
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    path_array = []
    global steps_done
    ##env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        path_array.append(data.dat())
        obs = env.reset()
        #print(f'obs {obs[0].size()}')
        state = get_state(obs[0],episode,False)
        path_array[episode].AddState(state.float())
        total_reward = 0.0
        for t in count():
            action = policy(state).max(1)[1].view(1,1)
            path_array[episode].AddAction(action)
            steps_done+=1
            if render:
                env.render()
                ##time.sleep(0.02)

            obs, reward, done, info,_ = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs,episode,False)
                path_array[episode].AddState(next_state.float())
            else:
                next_state = None

            state = next_state
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break
    
    env.close()
    print("-------------------------------\n L =",len( path_array[0].get_state())) #print state lenth
    return path_array

if __name__ == '__main__':
    tt = False
    #tt = True
    if(tt):
        optics.optic().clustering_from_mem()
        exit
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
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
    # create environment
    #env = gym.make("ALE/Alien-v5",render_mode="human")
    env = gym.make("ALE/Alien-v5")
    env=make_env(env)
    # create networks
    policy_net = DQN(n_actions=env.action_space.n).to(device)
    target_net = DQN(n_actions=env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    #train(env, 30001)
    #torch.save(policy_net, "dqn_alien_model_30001")
    
    policy_net = torch.load("dqn_alien_model_30001", map_location=torch.device('cpu'))
    path_array= test(env,40, policy_net, render=False)
    dis_graph = levenshtein_distance.PathDistanceCalculator().calculate_distances(path_array)
    print(dis_graph)
    print("----------------------------------------------")
    path = 'test.mem'
    f = open(path, 'w')
    for i in dis_graph:
        for j in i:
            f.write(str(int(j))+ " ")
        f.write("\n")
    f.close()
    optics.optic().clustering(dis_graph)
