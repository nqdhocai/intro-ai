import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

class QNet(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, num_actions, lr=1e-3, gamma=0.99):
        self.q_net = QNet(state_dim, num_actions)
        self.target_net = QNet(state_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.num_actions = num_actions
        self.buffer = []
        self.batch_size = 64
        self.epsilon = 0.1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return int(q_values.argmax())

    def store(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target = rewards + self.gamma * next_q

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())



# dqn_agent = DQNAgent(state_dim=env.state_dim, num_actions=env.action_space.n)

# losses = []
# rewards = []

# for epoch in range(1000):
#     state = env.reset()
#     action = dqn_agent.select_action(state)
#     next_state, reward, done, info = env.step(action)

#     dqn_agent.store((state, action, reward, next_state))
#     loss = dqn_agent.train_step()

#     if loss is not None:
#         losses.append(loss)
#     rewards.append(reward)

#     if epoch % 10 == 0:
#         dqn_agent.update_target()
#         print(f"[Epoch {epoch}]  Avg loss: {np.mean(losses[-10:]):.4f}  |  Avg reward: {np.mean(rewards[-10:]):.4f}")

# fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# ax[0].plot(losses)
# ax[0].set_title("Loss over time")
# ax[0].set_ylabel("Loss")

# ax[1].plot(rewards)
# ax[1].set_title("Reward over time")
# ax[1].set_ylabel("Reward")
# ax[1].set_xlabel("Epoch")

# plt.tight_layout()
# plt.show()