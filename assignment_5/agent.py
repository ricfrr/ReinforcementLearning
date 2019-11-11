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
        # standart initialization to 5.0
        #self.sigma = torch.tensor([5.0])  # TODO: Implement accordingly (T1, T2) 
        self.sigma_0 = torch.tensor([10.0]) # value of sigma_0
        #self.sigma = self.sigma_0 # T2 
        self.sigma = torch.nn.Parameter(torch.tensor([10.0])) # sigma as a parameter for task 2.2 
        
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
        sigma = F.softplus(self.sigma) 

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        normal_dist = Normal(mu, torch.sqrt(sigma))
        return normal_dist
        # TODO: Add a layer for state value calculation (T3)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.episode = 1
        self.C = -5*0.0001 # C parameter of the decay update
        self.action_probs = []
        self.rewards = []

    def episode_finished(self):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards,self.gamma) # computing the discounted reward 
        # normalize the discounted rewards task 1.c
        discounted_rewards -= torch.mean(discounted_rewards) 
        discounted_rewards /= torch.std(discounted_rewards)

        # TODO: Compute critic loss and advantages (T3)

        # TODO: Compute the optimization term (T1, T3)
        #weighted_probs = -action_probs * (discounted_rewards -20) # with baseline task 1.b 
        weighted_probs = -action_probs * (discounted_rewards) # without baseline
        
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        #computing the loss 
        loss = torch.mean(weighted_probs)
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # sigma implementation 
        #if self.episode % 100==0:
        #    print("sigma : "+str(self.policy.sigma))
        #sigma update for task 2.1
        #self.policy.sigma = self.policy.sigma_0 * np.exp(self.C*self.episode) # decay sigma update
        #self.episode += 1

        

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        # TODO: Pass state x through the policy network (T1)
        n_dist = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = n_dist.mean #mean of the distribution for evalueation 
        else:
            action = n_dist.sample((1,))[0] # sampling from the disctribution
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = n_dist.log_prob(action)
        # TODO: Return state value prediction, and/or save it somewhere (T3)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

