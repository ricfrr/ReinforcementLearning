import gym
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn


def discretize(state,x_grid,y_grid,theta_grid, x_dot_grid, y_dot_grid, theta_dot_grid):
    
    n_s = []
    n_s.append(np.digitize(state[0],x_grid,right=False))
    n_s.append(np.digitize(state[1],y_grid,right=False))
    n_s.append(np.digitize(state[2],theta_grid,right=False))
    n_s.append(np.digitize(state[3],x_dot_grid,right=False))
    n_s.append(np.digitize(state[4],y_dot_grid,right=False))
    n_s.append(np.digitize(state[5],theta_dot_grid,right=False))
    n_s.append(int(state[6]))
    n_s.append(int(state[7]))
    return n_s


np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 20
num_of_actions = 4

contact_value = 2

# Reasonable values for Cartpole discretization
discr = 16
""" x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4 """

# For LunarLander, use the following values:
#       [  x     y     xdot ydot theta  thetadot cl  cr
s_min = [ -1.2, -0.3, -2.4,  -2, -6.28,  -8, 0,   0 ]
s_max = [  1.2,  1.2,  2.4,   2,  6.28,   8, 1,   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 2223  # TODO: Set the correct value. simply solve the equation
initial_q = 0  # T3: Set to 50
initial_q = 50  # T3


# Create discretization grid
x_grid = np.linspace(s_min[0], s_max[0], discr-1)
y_grid = np.linspace(s_min[1], s_max[1], discr-1)
theta_grid = np.linspace(s_min[4], s_max[4], discr-1)
x_dot_grid = np.linspace(s_min[2], s_max[2], discr-1)
y_dot_grid = np.linspace(s_min[3], s_max[3], discr-1)
theta_dot_grid = np.linspace(s_min[5], s_max[5], discr-1)



q_grid = np.zeros((discr, discr, discr, discr, discr,discr, contact_value,contact_value, num_of_actions)) + initial_q


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, rw = env.reset(), False, 0
    
    
    epsilon =  0.2 
    #epsilon = a/(a+ep) #GLIE implementation
    while not done:
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        s = discretize(state,x_grid,y_grid,theta_grid,x_dot_grid,y_dot_grid,theta_dot_grid)
        #choose random action with probability 1-epsilon
        action = np.random.choice(
            [0,1,np.argmax(q_grid[s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7]])], 
            1,
            p=[epsilon/2, epsilon/2, 1-epsilon]
            )[0]  
        
        new_state, reward, done, _ = env.step(action)
        s_n =  discretize(new_state,x_grid,y_grid,theta_grid,x_dot_grid,y_dot_grid,theta_dot_grid)
        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            q_grid[s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],action]=  q_grid[s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],action] + alpha*(reward+gamma*np.max(q_grid[s_n[0],s_n[1],s_n[2],s_n[3],s_n[4],s_n[5],s_n[6],s_n[7]]) - q_grid[s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],action])
        else:
            env.render() 
            
        state = new_state
        rw += reward
    ep_lengths.append(rw)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average reward: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

# Save the Q-value array
np.save("q_values_lander.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
#values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
#np.save("value_func_lander.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
""" heat_map = np.zeros((discr, discr)) #x and theta 
for i in range(discr):
    for j in range(discr):
        heat_map[i,j] = np.mean(q_grid[i,:,j,:])

seaborn.heatmap(heat_map)
plt.xlabel('theta')
plt.ylabel('position')
plt.show() """

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode reward", "500 episode average"])
plt.title("Episode rewards")
plt.savefig("plots/Lander/train_reward_static_true.jpg")
plt.show()

