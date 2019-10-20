import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn


def discretize(state,x_grid,v_grid,th_grid,av_grid):
    
    n_s = []
    n_s.append(np.digitize(state[0],x_grid,right=False))
    n_s.append(np.digitize(state[1],v_grid,right=False))
    n_s.append(np.digitize(state[2],th_grid,right=False))
    n_s.append(np.digitize(state[3],av_grid,right=False))
    
    return n_s


np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 10000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 2223  # TODO: Set the correct value. simply solve the equation
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr-1)
v_grid = np.linspace(v_min, v_max, discr-1)
th_grid = np.linspace(th_min, th_max, discr-1)
av_grid = np.linspace(av_min, av_max, discr-1)


q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    
    
    epsilon =  0.2 # constant T3: Set to 0
    #epsilon = a/(a+ep) #GLIE implementation
    while not done:
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        s = discretize(state,x_grid,v_grid,th_grid,av_grid)
        #choose random action with probability 1-epsilon
        action = np.random.choice(
            [0,1,np.argmax(q_grid[s[0],s[1],s[2],s[3]])], 
            1,
            p=[epsilon/2, epsilon/2, 1-epsilon]
            )[0]  #int(np.random.rand()*2
        
        new_state, reward, done, _ = env.step(action)
        s_n = discretize(new_state,x_grid,v_grid,th_grid,av_grid)
        print(new_state)
        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            
            q_grid[s[0],s[1],s[2],s[3],action]=  q_grid[s[0],s[1],s[2],s[3],action] + alpha*(reward+gamma*np.max(q_grid[s_n[0],s_n[1],s_n[2],s_n[3]]) - q_grid[s[0],s[1],s[2],s[3],action])
            pass
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
heat_map = np.zeros((discr, discr)) #x and theta 
for i in range(discr):
    for j in range(discr):
        heat_map[i,j] =np.mean(q_grid[i,j])

seaborn.heatmap(heat_map)
plt.show()
#plt.imshow(heat_map)
#plt.show()
# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

