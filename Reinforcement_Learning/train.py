# train.py

import numpy as np
import os
from collections import deque
from rl_tutor.environment import RocketPropulsionEnv
from rl_tutor.agent import PPOAgent
from rl_tutor import config, utils

def main():
    """Main training loop for the simplified graph traversal task."""
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
        
    graph = utils.create_concept_graph()
    utils.visualize_graph(graph)

    env = RocketPropulsionEnv(graph)
    
    # Calculate dimensions and initialize agent
    state_dim = len(config.TOPICS) * 2
    action_dim = len(config.TOPICS)
    agent = PPOAgent(state_dim, action_dim)

    print("🚀 Starting training for RL Tutor Agent (Graph Coverage Task)...")

    scores_deque = deque(maxlen=100)
    mastered_topics_deque = deque(maxlen=100)
    
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []

    timestep = 0
    for e in range(1, config.N_EPISODES + 1):
        state = env.reset()
        state = np.reshape(state, [1, state_dim])
        
        episode_score = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            timestep += 1
            
            valid_actions = env._get_askable_actions()
            if not valid_actions:
                break
                
            action, log_prob = agent.act(state, valid_actions)
            
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_dim])

            states.append(state[0])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state[0])
            dones.append(done)
            log_probs.append(log_prob)

            state = next_state
            episode_score += reward

            if timestep % config.UPDATE_TIMESTEPS == 0 and len(states) > 0:
                agent.learn(states, actions, rewards, next_states, dones, log_probs)
                states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []

            if done:
                break
        
        scores_deque.append(episode_score)
        mastered_topics_deque.append(len(env.mastered_topics_in_episode))
        
        if e % 50 == 0 or e == 1:
            avg_score = np.mean(scores_deque)
            avg_mastered = np.mean(mastered_topics_deque)
            print(f"Ep {e}/{config.N_EPISODES} | Avg Score: {avg_score:.2f} | Avg Mastered: {avg_mastered:.1f}/{action_dim}")

    agent.save()
    print("\nTraining complete. Model saved.")

if __name__ == "__main__":
    main()

