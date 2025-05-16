from eval_agents import eval_baselines, eval_dqn, eval_ppo
from prepair_all import MODEL_IDS, dqn_agents, envs, ppo_agents, test_dataset
from train_agents import train_dqn_agent, train_ppo_agent

if __name__ == "__main__":
    for model_name in MODEL_IDS.keys():
        print(f"Training DQN agent for {model_name}...")
        train_dqn_agent(dqn_agents[model_name], envs[model_name])

        print(f"Training PPO agent for {model_name}...")
        train_ppo_agent(ppo_agents[model_name], envs[model_name])

        print(f"Evaluating DQN agent for {model_name}...")
        dqn_recalls, dqn_k_diff, dqn_full = eval_dqn(
            envs[model_name],
            dqn_agents[model_name],
            test_dataset,
            envs[model_name].k_candidates,
        )
        print(f"DQN Recall: {dqn_recalls}")
        print(f"DQN K Diff: {dqn_k_diff}")
        print(f"DQN Full: {dqn_full}")

        print(f"Evaluating PPO agent for {model_name}...")
        ppo_recalls, ppo_k_diff, ppo_full = eval_ppo(
            envs[model_name],
            ppo_agents[model_name],
            test_dataset,
            envs[model_name].k_candidates,
        )
        print(f"PPO Recall: {ppo_recalls}")
        print(f"PPO K Diff: {ppo_k_diff}")
        print(f"PPO Full: {ppo_full}")

    print("Evaluating baselines...")
    eval_baselines(envs[model_name], test_dataset, envs[model_name].k_candidates)
