# rl_tutor/environment.py

import numpy as np
from . import config, utils

class RocketPropulsionEnv:
    """
    Manages the student's knowledge state (the heatmap) and rewards.
    """
    def __init__(self, graph):
        self.graph = graph
        self.topic_map = {i: topic for i, topic in enumerate(config.TOPICS)}
        
        self.prereq_indices = {
            i: [config.TOPICS.index(p) for p, _ in self.graph.in_edges(topic)]
            for i, topic in self.topic_map.items()
        }
        
        self.state = np.zeros(config.STATE_SIZE)
        self._true_mastery = np.zeros(config.STATE_SIZE)
        self.mastered_topics_in_episode = set()

    def _get_valid_actions(self):
        """Returns a list of topic indices that can be asked about."""
        valid_actions = []
        for i in range(config.STATE_SIZE):
            if i in self.mastered_topics_in_episode:
                continue

            prereqs = self.prereq_indices.get(i, [])
            if not prereqs or all(self.state[p_idx] >= config.PROFICIENCY_THRESHOLD for p_idx in prereqs):
                valid_actions.append(i)
        
        if not valid_actions:
            valid_actions = [i for i in range(config.STATE_SIZE) if i not in self.mastered_topics_in_episode]
        
        return valid_actions

    def reset(self):
        """Resets the environment for a new student/episode."""
        self.state = np.zeros(config.STATE_SIZE)
        self._true_mastery = np.random.uniform(0.2, 0.8, config.STATE_SIZE)
        self.mastered_topics_in_episode = set()
        return self.state

    def step(self, action_idx):
        """Executes one time step: ask a question and get feedback."""
        prob_correct = self._true_mastery[action_idx]
        is_correct = np.random.random() < prob_correct

        old_state = np.copy(self.state)
        
        if is_correct:
            self.state[action_idx] += 0.25
        else:
            self.state[action_idx] -= 0.10
        
        self.state = np.clip(self.state, 0, 1)
        
        # --- NEW REWARD LOGIC ---
        reward = 0
        
        # 1. First-Time Mastery Bonus
        if old_state[action_idx] < config.MASTERY_THRESHOLD and self.state[action_idx] >= config.MASTERY_THRESHOLD:
            reward += config.FIRST_MASTERY_BONUS
            self.mastered_topics_in_episode.add(action_idx)

        # 2. Base reward from proficiency gain (and cost)
        proficiency_gain = np.sum(self.state) - np.sum(old_state)
        reward += proficiency_gain - config.ASKING_COST

        # 3. Check for episode completion
        mastery_ratio = len(self.mastered_topics_in_episode) / config.STATE_SIZE
        done = mastery_ratio >= config.COMPLETION_THRESHOLD

        # 4. Completion Bonus
        if done:
            reward += config.COMPLETION_BONUS
        
        return self.state, reward, done, {}

