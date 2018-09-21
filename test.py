from unityagents import UnityEnvironment
from dqn_agent import Agent
import numpy as np
import sys
import torch
from collections import deque
import matplotlib.pyplot as plt

# Check for model filename argument
model_file = 'checkpoint.pth'
if (len(sys.argv) > 1):
    model_file = sys.argv[1]
print ("Using model ",model_file)

plt.ion()

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of actions and state size
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)
    
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# Load the weights from file
agent.qnetwork_local.load_state_dict(torch.load(model_file))

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = agent.act(state).item()               # item() converts form numpy type to native integer
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))
env.close()