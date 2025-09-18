# rl_tutor/config.py

import os

# =============================================================================
# GRAPH & TOPIC DEFINITION
# =============================================================================
# The core knowledge graph for the rocket propulsion curriculum.
TOPICS = [
    "Conservation of Momentum", "Ideal Gas Law", "Basic Thermodynamics", "Heat Transfer",
    "Newton's Third Law", "Propellant Chemistry", "Nozzle Theory", "Combustion Chamber Design",
    "Types of Propellants", "Thrust Calculation", "Atmospheric Drag", "Specific Impulse (Isp)",
    "Rocket Staging", "Tsiolkovsky Rocket Equation", "Orbital Mechanics"
]

DEPENDENCIES = [
    ("Conservation of Momentum", "Newton's Third Law"), ("Ideal Gas Law", "Basic Thermodynamics"),
    ("Basic Thermodynamics", "Nozzle Theory"), ("Heat Transfer", "Nozzle Theory"),
    ("Heat Transfer", "Combustion Chamber Design"), ("Newton's Third Law", "Thrust Calculation"),
    ("Propellant Chemistry", "Combustion Chamber Design"), ("Propellant Chemistry", "Types of Propellants"),
    ("Nozzle Theory", "Thrust Calculation"), ("Combustion Chamber Design", "Thrust Calculation"),
    ("Thrust Calculation", "Atmospheric Drag"), ("Thrust Calculation", "Specific Impulse (Isp)"),
    ("Specific Impulse (Isp)", "Tsiolkovsky Rocket Equation"), ("Rocket Staging", "Tsiolkovsky Rocket Equation"),
    ("Thrust Calculation", "Rocket Staging"), ("Tsiolkovsky Rocket Equation", "Orbital Mechanics")
]


# =============================================================================
# ENVIRONMENT & REWARD SETTINGS
# =============================================================================
# The goal is to master this many topics to complete an episode.
COMPLETION_TARGET = 14
# Proficiency required in prerequisites to unlock a new topic.
PROFICIENCY_THRESHOLD = 0.8
# Proficiency required to consider a topic "mastered".
MASTERY_THRESHOLD = 0.9
# Maximum number of questions before an episode terminates.
MAX_STEPS_PER_EPISODE = 30

# --- Reward Structure ---
ASKING_COST = 1.0                   # Penalty for each question asked.
COMPLETION_BONUS = 100.0            # Large bonus for finishing the graph.
FIRST_MASTERY_BONUS = 5.0           # Bonus for mastering a topic for the first time.
UNLOCK_BONUS = 10.0                 # Bonus for unlocking new topics.
REPEATED_QUESTION_PENALTY = 2.0     # Penalty for asking about an already-mastered topic.


# =============================================================================
# PPO AGENT HYPERPARAMETERS
# =============================================================================
NETWORK_SIZE = 256                  # Hidden layer size for Actor and Critic networks.
GAMMA = 0.99                        # Discount factor for future rewards.
GAE_LAMBDA = 0.95                   # Lambda for Generalized Advantage Estimation.
CLIP_EPSILON = 0.2                  # Epsilon for PPO clipping.
ACTOR_LEARNING_RATE = 3e-4          # Learning rate for the policy network (Actor).
CRITIC_LEARNING_RATE = 1e-3         # Learning rate for the value network (Critic).
ENTROPY_BETA = 0.01                 # Coefficient for the entropy bonus to encourage exploration.


# =============================================================================
# TRAINING SETTINGS
# =============================================================================
N_EPISODES = 3000                   # Total number of episodes to train for.
UPDATE_TIMESTEPS = 2048             # Number of steps to collect before a PPO update.
EPOCHS_PER_UPDATE = 10              # Number of optimization epochs per PPO update.


# =============================================================================
# FILE & DIRECTORY PATHS
# =============================================================================
MODEL_DIR = "models"
GRAPH_IMAGE_PATH = "concept_graph.png"
INFERENCE_GRAPH_PATH = "inference_path_heatmap.png"

