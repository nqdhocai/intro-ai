import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2, entropy_coeff=0.01):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, _ = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), dist.entropy().item()

    def store(self, transition):
        self.memory.append(transition)  # (state, action, log_prob, reward, value, done)

    def train(self, epochs=4, batch_size=64):
        if len(self.memory) == 0:
            return

        # Unpack memory
        states, actions, log_probs, rewards, values, dones = zip(*self.memory)
        self.memory = []

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        returns = self.compute_returns(rewards, dones, values)
        advantages = returns - values

        all_losses = []
        for _ in range(epochs):
            logits, new_values = self.model(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_losses.append(loss.item())
        return np.mean(all_losses)

    def compute_returns(self, rewards, dones, values):
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns).to(self.device)

# ppo_agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_space.n)

# total_rewards = []
# total_losses = []
# action_counts = np.zeros(env.action_space.n)

# for epoch in range(1000):
#     state = env.reset()
#     done = False

#     action, log_prob, entropy = ppo_agent.select_action(state)
#     next_state, reward, done, info = env.step(action)

#     _, value = ppo_agent.model(torch.FloatTensor(state).unsqueeze(0).to(ppo_agent.device))
#     ppo_agent.store((state, action, log_prob, reward, value.item(), done))

#     total_rewards.append(reward)
#     action_counts[action] += 1

#     if (epoch + 1) % 32 == 0:
#         ppo_agent.train()  # gọi train ở đây sẽ reset memory
#         print(f"[Epoch {epoch+1}] Loss avg: {np.mean(total_losses[-32:]):.4f} | Reward avg: {np.mean(total_rewards[-32:]):.4f} | Action dist: {action_counts}")
#         action_counts[:] = 0



# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(total_rewards)
# plt.title("Reward per episode")
# plt.xlabel("Episode")
# plt.ylabel("Reward")

# plt.subplot(1, 2, 2)
# plt.plot(total_losses)
# plt.title("Average PPO loss every 32 steps")
# plt.xlabel("Train step")
# plt.ylabel("Loss")
# plt.tight_layout()

# plt.bar(range(env.action_space.n), action_counts)
# plt.xticks(range(env.action_space.n), env.k_candidates)
# plt.title("Action (k) distribution in last window")
# plt.xlabel("k value")
# plt.ylabel("Count")

# fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# ax[0].plot(losses)
# ax[0].set_title("Loss over time")
# ax[0].set_ylabel("Loss")

# ax[1].plot(rewards)
# ax[1].set_title("Reward over time")
# ax[1].set_ylabel("Reward")
# ax[1].set_xlabel("Epoch")

# plt.show()