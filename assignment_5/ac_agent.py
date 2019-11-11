import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards
import torch.distributions 


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_val = torch.nn.Linear(self.hidden, 1)
        self.sigma = torch.nn.Parameter(torch.tensor([10.0])) # sigma learned automatically
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        # go also the other critic part
        val = self.fc2_val(x)
       
        sigma = self.sigma  # TODO: Is it a good idea to leave it like this?

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        normal_dist = Normal(mu, sigma ) # 
        return normal_dist , val 
        # TODO: Add a layer for state value calculation (T3)



class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = [] # values saved during the training 
        self.next_values = [] # save if and episode is the last one or not 

    def episode_finished(self):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1) # values from the network
        next_values = torch.stack(self.next_values, dim=0).to(self.train_device).squeeze(-1) # values from the network
        self.states, self.action_probs, self.rewards, self.values , self.next_values= [], [], [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        
        # discounted_rewards = discount_rewards(rewards,self.gamma) # TASK 3
        
        # TODO: Compute critic loss and advantages (T3)
        # compute the advantage 
        #advantage =  discounted_rewards - values # advantage TASK 3
        #advantage -= torch.mean(advantage)
        #advantage /= torch.std(advantage.detach())

        #TASK 4 advantage using TD(0)
        advantage = rewards + self.gamma * next_values - values  
        

        weighted_probs = -action_probs* advantage.detach() 

        # compute the actor and the critic loss
        # compute the loss
        actor_loss = weighted_probs.mean()
        critic_loss = advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss 
        
        
        ac_loss.backward()

    
        # TODO: Compute the optimization term (T1, T3)
        self.optimizer.step()
        self.optimizer.zero_grad()

        

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        n_dist, value = self.policy.forward(x)
        
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = n_dist.mean
        else:
            action = n_dist.sample()
        #print("action inside action " + str(action))
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = n_dist.log_prob(action)
        # TODO: Return state value prediction, and/or save it somewhere (T3)
        # append the value prediction to the agent 
        self.values.append(value)
        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        
    
    def store_next_values(self , v_next):
        self.next_values.append(v_next)

