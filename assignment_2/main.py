import numpy as np
import copy
from time import sleep
from sailing import SailingGridworld


# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

def treshold_vi(m_1, m_2, epsilon= 1e-4):
    for i in range(0,m_1.shape[0]):
        for j in range(0,m_1.shape[1]):
            if (np.abs(m_1[i,j]-m_2[i,j]) >= epsilon):
                return False

    return True

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
   

    # Code for the TASK 1 and 2, the policy update is done inside the loop
    # there is also a control to check when the policy and value function converge for the Q4
    
    #TODO policy creation
    #loop until the treshold is reached
    #task 1 implementation 4 four nested loop for looping trough all the possible action
    '''
    for j in range(0,100):
        old_policy = copy.deepcopy(policy)
        old_value = copy.deepcopy(value_est)
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
                policy[i,k] = m_index #update the policy with the action that has the maximum value
        # check when policy and value function converge
    
        if np.array_equal(old_policy,policy):
            print ("policy equal at: "+str(j)+" episode")
        if np.array_equal(old_value,value_est):
            print ("value equal at: "+str(j) +" episode")
    '''    
    
    ## TASK 3 loop modified with the treshold
      
    #loop until the treshold is reached
    j=0
    while  True:
        j+=1
        old_value = copy.deepcopy(value_est)
        for i in range(0,env.w): # loop trough all the cells
            for k in range(0,env.h):
                act = np.zeros(4) # problem here
                m_v =-100000
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
                policy[i,k] = m_index #update the policy with the action that has the maximum value

        if treshold_vi(old_value,value_est):
            print ("converge at : "+str(j))
            break
              
    ## end TASK 4
    
   

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
    N = 1000 # number of iteration 
    G = [] # vector of discounted reward

    # multiple episode running for computing the discounted return 
    for j in range(0,N):
        env.reset()
        # if the episode if finished
        done = False
        disc_ret =0
        r_v = []
        while not done:
            action = policy[state[0],state[1]]
            state, reward, done, _ = env.step(action)
            r_v.append(reward)
        k =0
        disc_ret = 0.9**(len(r_v)-1)*r_v[len(r_v)-1] # discounted return, final reward because all the other rewards are zero
 
        G.append(disc_ret)
        print("episode : "+str(j)+" done")
    
    print("mean :"+ str(np.mean(G)))
    print("std dev :"+ str(np.std(G)))


    '''
    done = False
    while not done:
        # Select a random action
        # TODO: Use the policy to take the optimal action (Task 2)
        action = policy[state[0],state[1]]

        # Step the environment
        state, reward, done, _ = env.step(action)

        # Render and sleep
        env.render()
        sleep(0.1)
    '''


