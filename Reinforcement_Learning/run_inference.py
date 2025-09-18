# run_inference.py

import numpy as np
import torch
from rl_tutor.environment import RocketPropulsionEnv
from rl_tutor.agent import PPOAgent
from rl_tutor import config, utils

def main():
    """Runs the trained agent to generate an optimal topic coverage path."""
    graph = utils.create_concept_graph()

    env = RocketPropulsionEnv(graph)
    
    # Calculate dimensions and initialize agent
    state_dim = len(config.TOPICS) * 2
    action_dim = len(config.TOPICS)
    agent = PPOAgent(state_dim, action_dim)

    try:
        agent.load()
        print("\nModels loaded successfully.")
    except FileNotFoundError:
        print("\nERROR: Trained models not found. Please run train.py first.")
        return

    print("\n🤖 Running inference to find optimal topic coverage path...")
    
    state = env.reset()
    coverage_path = []
    topic_counts = {i: 0 for i in range(action_dim)}

    for _ in range(config.MAX_STEPS_PER_EPISODE):
        state_np = np.reshape(state, [1, state_dim])
        valid_actions = env._get_askable_actions()
        if not valid_actions:
            break
            
        with torch.no_grad():
            action, _ = agent.act(state_np, valid_actions)
        
        coverage_path.append(config.TOPICS[action])
        topic_counts[action] += 1
        
        state, _, done, _ = env.step(action)
        if done:
            break
            
    print("\n✨ Optimal Topic Coverage Path Generated ✨")
    print(" -> ".join(coverage_path))

    print("\nFinal Proficiency Heatmap:")
    for i, prof in enumerate(env.proficiency):
        print(f"- {config.TOPICS[i]:<28}: {prof:.2f}")

    utils.visualize_inference_path(graph, topic_counts)
    print(f"\nInference path graph saved to {config.INFERENCE_GRAPH_PATH}")

if __name__ == "__main__":
    main()

