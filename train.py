""" A script for training a DQN agent in the Banana environment of Udacity Deep Reinforcement Learning nanodegree (Project 1).
Learning statistics will be printed in the standard output; a plot of the progress will be saved in file training.png.
After completion, neural network weights will be saved in file checkpoint.pth. 
This code is heavily based on the proposed exercise, with small changes for compatibility with the Unity environment, and use in the command-line.
"""

from unityagents import UnityEnvironment
from dqn_agent import Agent
import numpy as np
import sys 
import torch
from collections import deque
import matplotlib.pyplot as plt

use_ddqn = True
if (len(sys.argv) > 1):
    if(sys.argv[1]=='dqn'):
        use_ddqn=False
        print("Using vanilla DQN instead of Double DQN.")
    else:
        print("Using Double DQN for training. Pass dqn as argument to choose vanilla DQN.")

plt.ion()

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions and state size
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

# Instantate the agent, using DQN or Double DQN according to the command line argument
agent = Agent(state_size=state_size, action_size=action_size, seed=0, use_ddqn=use_ddqn)

def dqn(n_episodes=2000, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]   # initial state

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps).item() # item() converts form numpy type to native integer
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.show()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('training.png')

env.close()