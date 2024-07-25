import numpy as np
import torch
import gym
from sklearn.cluster import OPTICS

class CombinedEditDistance:
    def __init__(self, state_weight=0.5, action_weight=0.5):    #權重
        self.state_weight = state_weight
        self.action_weight = action_weight

    def state_distance(self, state1, state2):
        # state (1, 4, 84, 84)
        # 均方誤差
        return torch.nn.functional.mse_loss(state1, state2, reduction='sum').item()

    def action_distance(self, action1, action2):
        # action (1, 1) 
        return 0 if action1.item() == action2.item() else 1

    def calculate(self, path1, path2):
        states1 = path1.state_array
        states2 = path2.state_array
        actions1 = path1.action_array
        actions2 = path2.action_array

        m, n = len(actions1), len(actions2)
        dp = np.zeros((m+1, n+1))

        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j

        for i in range(1, m+1):
            for j in range(1, n+1):
                state_dist = self.state_distance(states1[i-1], states2[j-1])
                action_dist = self.action_distance(actions1[i-1], actions2[j-1])
                
                combined_dist = (self.state_weight * state_dist + 
                                 self.action_weight * action_dist)
                combined_dist//=int(1e6) 
                #print("comb",combined_dist)
                dp[i][j] = min(dp[i-1][j] + 2,          # 刪除
                               dp[i][j-1] + 2,          # 插入
                               dp[i-1][j-1] + combined_dist)  # 替換或匹配

        return dp[m][n]

class PathDistanceCalculator:
    def __init__(self, state_weight=0.5, action_weight=0.5):
        self.edit_distance = CombinedEditDistance(state_weight, action_weight)
    
    def calculate_distances(self, path_array):
        n = len(path_array)
        distance_matrix = np.zeros((n, n))
        '''
        for i in range(len( path_array)):
            for j in range (len( path_array[i].state_array)):
                path_array[i].state_array[j] = path_array[i].state_array[j].float()
        '''
        for i in range(n):
            for j in range(i+1, n):
                print("caculate",i,j)
                dist = self.edit_distance.calculate(path_array[i], path_array[j])
                distance_matrix[i, j] = dist*5
                distance_matrix[j, i] = dist*5
        
        return distance_matrix
