# rl_tutor/environment.py

import numpy as np
import networkx as nx
from . import config

class RocketPropulsionEnv:
    """
    A simplified environment where the goal is to cover all topics in the graph.
    There is no student simulation; asking about a topic increases proficiency.
    """
    def __init__(self, graph):
        self.graph = graph
        self.topics = config.TOPICS
        self.num_topics = len(self.topics)
        self.topic_map = {i: topic for i, topic in enumerate(self.topics)}

        self.prereq_indices = {
            i: [self.topics.index(p) for p, _ in self.graph.in_edges(topic)]
            for i, topic in self.topic_map.items()
        }
        self.state = self.reset()

    def _get_unlock_mask(self):
        """Determines which topics are available based on prerequisite proficiency."""
        unlock_mask = np.zeros(self.num_topics)
        for i in range(self.num_topics):
            prereqs = self.prereq_indices.get(i, [])
            if not prereqs or all(self.proficiency[p_idx] >= config.PROFICIENCY_THRESHOLD for p_idx in prereqs):
                unlock_mask[i] = 1
        return unlock_mask

    def _get_askable_actions(self):
        """Returns a list of topic indices that are currently unlocked."""
        return [i for i, unlocked in enumerate(self.unlock_mask) if unlocked]

    def reset(self):
        """Resets the environment for a new episode."""
        self.proficiency = np.zeros(self.num_topics)
        self.unlock_mask = self._get_unlock_mask()
        self.mastered_topics_in_episode = set()
        self.step_count = 0
        
        # State is now just proficiency and the unlock mask
        return np.concatenate([self.proficiency, self.unlock_mask])

    def step(self, action_idx):
        """
        Executes one step. "Asking" about a topic deterministically increases its proficiency.
        """
        self.step_count += 1
        
        # --- REWARD CALCULATION ---
        reward = -config.ASKING_COST
        
        # Penalty for re-visiting a mastered topic
        if self.proficiency[action_idx] >= config.MASTERY_THRESHOLD:
            reward -= config.REPEATED_QUESTION_PENALTY
            
        # Deterministically increase proficiency for the chosen topic
        # It takes multiple visits to master a topic
        self.proficiency[action_idx] += 0.4
        self.proficiency[action_idx] = min(1.0, self.proficiency[action_idx])

        # Check for first-time mastery of the topic
        if self.proficiency[action_idx] >= config.MASTERY_THRESHOLD and action_idx not in self.mastered_topics_in_episode:
            self.mastered_topics_in_episode.add(action_idx)
            reward += config.FIRST_MASTERY_BONUS

        # Check if this action unlocked new topics
        old_unlock_mask = self.unlock_mask
        self.unlock_mask = self._get_unlock_mask()
        num_newly_unlocked = np.sum(self.unlock_mask) - np.sum(old_unlock_mask)
        if num_newly_unlocked > 0:
            reward += num_newly_unlocked * config.UNLOCK_BONUS
            
        # Check for completion of the entire graph
        done = False
        if len(self.mastered_topics_in_episode) >= config.COMPLETION_TARGET:
            reward += config.COMPLETION_BONUS
            done = True
        
        # End episode if it runs too long
        if self.step_count >= config.MAX_STEPS_PER_EPISODE:
            done = True

        self.state = np.concatenate([self.proficiency, self.unlock_mask])
        
        return self.state, reward, done, {}

