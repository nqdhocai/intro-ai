from prepair_all import ppo_agents, dqn_agents, envs, MODEL_IDS 
import numpy as np
import torch

def train_dqn_agent(dqn_agent, env):
    """
    Train the DQN agent.
    
    Parameters:
        dqn_agent: DQNAgent instance
        env: RAGTopKEnv instance
    """
    # Initialize variables to store losses and rewards
    losses = []
    rewards = []

    for epoch in range(1000):
        state = env.reset()
        action = dqn_agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        dqn_agent.store((state, action, reward, next_state))
        loss = dqn_agent.train_step()

        if loss is not None:
            losses.append(loss)
        rewards.append(reward)

        if epoch % 10 == 0:
            dqn_agent.update_target()
            print(f"[Epoch {epoch}]  Avg loss: {np.mean(losses[-10:]):.4f}  |  Avg reward: {np.mean(rewards[-10:]):.4f}")

def train_ppo_agent(ppo_agent, env):
    """
    Train the PPO agent.
    
    Parameters:
        ppo_agent: PPOAgent instance
        env: RAGTopKEnv instance
    """
    total_rewards = []
    total_losses = []
    action_counts = np.zeros(env.action_space.n)

    for epoch in range(1000):
        state = env.reset()
        done = False

        action, log_prob, entropy = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        _, value = ppo_agent.model(torch.FloatTensor(state).unsqueeze(0).to(ppo_agent.device))
        ppo_agent.store((state, action, log_prob, reward, value.item(), done))

        total_rewards.append(reward)
        action_counts[action] += 1

        if (epoch + 1) % 32 == 0:
            ppo_agent.train()  # gọi train ở đây sẽ reset memory
            print(f"[Epoch {epoch+1}] Loss avg: {np.mean(total_losses[-32:]):.4f} | Reward avg: {np.mean(total_rewards[-32:]):.4f} | Action dist: {action_counts}")
            action_counts[:] = 0

if __name__ == "__main__":
    
    for model_name in MODEL_IDS.keys():
        print(f"Training DQN agent for {model_name}...")
        train_dqn_agent(dqn_agents[model_name], envs[model_name])

        print(f"Training PPO agent for {model_name}...")
        train_ppo_agent(ppo_agents[model_name], envs[model_name])