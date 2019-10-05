import numpy as np
from time import sleep
from sailing import SailingGridworld


# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    
    #loop until the treshold is reached
    
    for j in range(0,100):
        for i in range(0,env.w): # loop trough all the cells
            for k in range(0,env.h):
                act = np.zeros(4) # problem here
                m_v =0
                m_index = 0
                for a in range(0,4): #loop trough possible action
                    for tr in env.transitions[i,k,a]:
                        if tr[0]== None:
                            continue                
                        act[a] += tr[3] * (tr[1] + 0.9 * value_est[tr[0][0],tr[0][1]])  
                    if act[a]>=m_v:
                        m_v= act[a]
                        m_index = a
                value_est[i,k] = m_v
                policy[i,k] = m_index
        
    
    #TODO policy creation
                

    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    done = False
    while not done:
        # Select a random action
        # TODO: Use the policy to take the optimal action (Task 2)
        action = policy[state[0],state[1]]

        # Step the environment
        state, reward, done, _ = env.step(action)

        # Render and sleep
        env.render()
        sleep(0.5)

