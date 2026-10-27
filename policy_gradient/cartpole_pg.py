import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # Simple feedforward neural network for policy
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # Output probabilities for actions
        )

    def forward(self, x):
        return self.fc(x)

def train_cartpole():
    # Setup environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    gamma = 0.95
    
    num_episodes = 500
    episode_rewards = []
    
    print("Starting CartPole-v1 Training...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        # Collect trajectory
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy_net(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            log_probs.append(m.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)
            
        episode_rewards.append(sum(rewards))
        
        # Calculate discounted returns (G_t)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Normalize returns for numerical stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
        # Policy Gradient Loss calculation
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G  # Gradient ascent on expected rewards
            
        # Optimize policy weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 50): {avg_reward:.2f}")
            
    env.close()
    return episode_rewards

if __name__ == "__main__":
    rewards = train_cartpole()
    
    # Calculate 100-episode moving average
    moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    
    # Plotting
    os.makedirs("../reports", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.4, color="blue")
    plt.plot(range(99, len(rewards)), moving_avg, label="100-Episode Moving Average", color="red", linewidth=2)
    plt.xlabel("Training Episodes")
    plt.ylabel("Total Reward")
    plt.title("Policy Gradient Training on CartPole-v1")
    plt.legend()
    plt.grid(True)
    plt.savefig("../reports/cartpole_training.png", dpi=300)
    print("CartPole plot saved to reports/cartpole_training.png")
    plt.show()