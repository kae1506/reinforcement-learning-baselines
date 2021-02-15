import numpy as np
import torch
import random

class ReplayBuffer():
    def __init__(self, maxSize, stateShape):
        self.memSize = maxSize
        self.memCount = 0

        self.stateMemory        = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.actionMemory       = np.zeros( self.memSize,               dtype=np.int64  )
        self.rewardMemory       = np.zeros( self.memSize,               dtype=np.float32)
        self.nextStateMemory    = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.doneMemory         = np.zeros( self.memSize,               dtype=np.bool   )
        self.stepMemory         = np.array( [          ],               dtype=np.int64  )

    def storeMemory(self, state, action, reward, nextState, done):
        memIndex = self.memCount % self.memSize

        self.stateMemory[memIndex]      = state
        self.actionMemory[memIndex]     = action
        self.rewardMemory[memIndex]     = reward
        self.nextStateMemory[memIndex]  = nextState
        self.doneMemory[memIndex]       = done
        self.stepMemory = np.append(self.stepMemory, self.memCount)

        self.memCount += 1


    def save(self, filename):
        if filename==None:
            filename = 'memory.mems'

        save_dict = {
            'state': self.stateMemory,
            'action': self.actionMemory,
            'reward': self.rewardMemory,
            'nextState': self.nextStateMemory,
            'done': self.doneMemory,
            'mem': self.memCount
        }

        torch.save(save_dict, filename)

    def load(self, filename):
        load_dict = torch.load(filename)

        self.stateMemory     = load_dict['state']
        self.actionMemory    = load_dict['action']
        self.rewardMemory    = load_dict['reward']
        self.nextStateMemory = load_dict['nextState']
        self.doneMemory      = load_dict['done']
        self.memCount        = load_dict['mem']

    def sample(self, sampleSize, weights=None):
        memMax = min(self.memCount, self.memSize)
        #print(len(weights), memMax)
        batchIndecies = random.choices(range(memMax), k=sampleSize, weights=weights)

        states      = self.stateMemory[batchIndecies]
        actions     = self.actionMemory[batchIndecies]
        rewards     = self.rewardMemory[batchIndecies]
        nextStates  = self.nextStateMemory[batchIndecies]
        dones       = self.doneMemory[batchIndecies]

        return states, actions, rewards, nextStates, dones
