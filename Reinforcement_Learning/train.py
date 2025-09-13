# train.py

import numpy as np
import os
from rl_tutor import config, utils
from rl_tutor.environment import RocketPropulsionEnv
from rl_tutor.agent import PPOAgent

def main():
    """Main function to run the PPO training loop."""
    graph = utils.create_concept_graph()
    utils.visualize_graph(graph)
    
    env = RocketPropulsionEnv(graph)
    agent = PPOAgent()
    
    print("🚀 Starting training for Rocket Propulsion Tutor Agent (PPO)...")

    total_timesteps = 0
    for e in range(config.N_EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, config.STATE_SIZE])
        episode_reward = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            total_timesteps += 1
            
            valid_actions = env._get_valid_actions()
            if not valid_actions: break

            action, prob = agent.act(state, valid_actions)
            if action is None: continue
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            agent.remember(state[0], action, prob, reward, next_state, done)
            state = np.reshape(next_state, [1, config.STATE_SIZE])
            
            if total_timesteps % config.UPDATE_TIMESTEPS == 0:
                agent.learn()

            if done: break
        
        print(f"Episode: {e+1}/{config.N_EPISODES}, Steps: {step+1}, Total Reward: {episode_reward:.2f}")

    agent.save()
    print("✅ Training complete.")

if __name__ == "__main__":
    main()

