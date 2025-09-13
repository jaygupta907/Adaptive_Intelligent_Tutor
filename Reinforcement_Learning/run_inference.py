# run_inference.py

import numpy as np
import os
from rl_tutor import config, utils
from rl_tutor.environment import RocketPropulsionEnv
from rl_tutor.agent import PPOAgent

def main():
    """Runs the trained PPO agent to generate an optimal learning path."""
    if not os.path.exists(config.ACTOR_MODEL_PATH):
        print(f"Error: Model not found at '{config.ACTOR_MODEL_PATH}'. Please run train.py first.")
        return

    graph = utils.create_concept_graph()
    env = RocketPropulsionEnv(graph)
    agent = PPOAgent()
    agent.load()

    print("\n🤖 Running inference with the trained PPO agent...")
    
    state = env.reset()
    path = []
    question_counts = {topic: 0 for topic in config.TOPICS}

    for _ in range(config.MAX_STEPS_PER_EPISODE):
        state_reshaped = np.reshape(state, [1, config.STATE_SIZE])
        valid_actions = env._get_valid_actions()
        if not valid_actions: break

        action, _ = agent.act(state_reshaped, valid_actions, force_exploit=True)
        
        topic_name = env.topic_map[action]
        path.append(topic_name)
        question_counts[topic_name] += 1
        
        state, _, done, _ = env.step(action)

        if done: break

    print("\n✨ Optimal Learning Path Generated ✨")
    print(" -> ".join(path))
    
    print("\nFinal Proficiency Heatmap:")
    for topic, proficiency in zip(config.TOPICS, state):
        print(f"- {topic:<30}: {proficiency:.2f}")

    # Generate and save the colored graph visualization
    utils.visualize_inference_path(graph, question_counts)

if __name__ == "__main__":
    main()

