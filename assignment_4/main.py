import gym
import numpy as np
from matplotlib import pyplot as plt
from rbf_agent import Agent as RBFAgent  # Use for Tasks 1-3
from dqn_agent import Agent as DQNAgent  # Task 4
from itertools import count
import torch
from utils import plot_rewards
import seaborn


def plot_2d(agent):    
    discr = 100
    x_min, x_max = -2.4, 2.4
    th_min, th_max = -0.3, 0.3
    x_grid = np.linspace(x_min, x_max, discr) 
    th_grid = np.linspace(th_min, th_max, discr)
    p_2d = np.zeros((discr, discr)) 
    for i in range(discr):
        for j in range(discr):
            #print(str(x_grid[i])+" : "+ str(th_grid[j]) )

            st = [x_grid[i],0,th_grid[j],0]
            st = np.array(st)
            p_2d[i,j] = agent.get_action(st)
    ax = seaborn.heatmap(p_2d)
    for ind, l in enumerate(zip(ax.get_xticklabels(),ax.get_yticklabels())):
        labelx, labely = l
        if ind % 10 == 0 or ind == discr-1:
            labelx.set_visible(True)
            labely.set_visible(True)
        else:
            labelx.set_visible(False)
            labely.set_visible(False)
    ax.set(xlabel='Theta',ylabel='Poistion')
    plt.show()

            



env_name = "CartPole-v0"
#env_name = "LunarLander-v2"
env = gym.make(env_name)
env.reset()

# Set hyperparameters
# Values for RBF (Tasks 1-3)
glie_a = 50
num_episodes = 10

# Values for DQN  (Task 4)
# if "CartPole" in env_name:
#     TARGET_UPDATE = 20
#     glie_a = 200
#     num_episodes = 5000
#     hidden = 12
#     gamma = 0.98
#     replay_buffer_size = 50000
#     batch_size = 32
# elif "LunarLander" in env_name:
#     TARGET_UPDATE = 20
#     glie_a = 5000
#     num_episodes = 15000
#     hidden = 64
#     gamma = 0.95
#     replay_buffer_size = 50000
#     batch_size = 128
# else:
#     raise ValueError("Please provide hyperparameters for %s" % env_name)


# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape[0]
print(state_space_dim)

# Tasks 1-3 - RBF
agent = RBFAgent(n_actions)

# Task 4 - DQN
# agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
#               hidden, gamma)

# Training loop
cumulative_rewards = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    done = False
    eps = glie_a/(glie_a+ep)
    cum_reward = 0
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        # Task 1: TODO: Update the Q-values
        #agent.single_update(state,action,next_state,reward,done)
        
        # Task 2: TODO: Store transition and batch-update Q-values
        agent.store_transition(state,action,next_state,reward,done)
        agent.update_estimator()
        # Task 4: Update the DQN

        # Move to the next state
        state = next_state
    cumulative_rewards.append(cum_reward)
    print("ep : "+ str(ep)+" rew : "+str(cum_reward))
    #plot_rewards(cumulative_rewards)

    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    # if ep % TARGET_UPDATE == 0:
    #     agent.update_target_network()

    # Save the policy
    # Uncomment for Task 4
    # if ep % 1000 == 0:
    #     torch.save(agent.policy_net.state_dict(),
    #               "weights_%s_%d.mdl" % (env_name, ep))
plot_2d(agent)
plot_rewards(cumulative_rewards)
print('Complete')
plt.ioff()
plt.show()

# Task 3 - plot the policy

