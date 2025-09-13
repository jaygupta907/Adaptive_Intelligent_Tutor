# rl_tutor/config.py

# --- GRAPH DEFINITION ---
TOPICS = [
    # Foundational Physics (Tier 0)
    "Conservation of Momentum",  # 0
    "Ideal Gas Law",             # 1
    "Basic Thermodynamics",      # 2
    "Heat Transfer",             # 3

    # Basic Rocket Principles (Tier 1)
    "Newton's Third Law",        # 4
    "Propellant Chemistry",      # 5
    
    # Core Components (Tier 2)
    "Nozzle Theory",             # 6
    "Combustion Chamber Design", # 7
    "Types of Propellants",      # 8

    # Performance Metrics (Tier 3)
    "Thrust Calculation",        # 9
    "Atmospheric Drag",          # 10
    "Specific Impulse (Isp)",    # 11

    # System Integration (Tier 4)
    "Rocket Staging",            # 12
    "Tsiolkovsky Rocket Equation",# 13
    "Orbital Mechanics"          # 14
]

DEPENDENCIES = [
    # Tier 0 to Tier 1
    ("Conservation of Momentum", "Newton's Third Law"),
    ("Ideal Gas Law", "Basic Thermodynamics"),
    ("Basic Thermodynamics", "Nozzle Theory"),
    ("Heat Transfer", "Nozzle Theory"),
    ("Heat Transfer", "Combustion Chamber Design"),

    # Tier 1 to Tier 2
    ("Newton's Third Law", "Thrust Calculation"),
    ("Propellant Chemistry", "Combustion Chamber Design"),
    ("Propellant Chemistry", "Types of Propellants"),

    # Tier 2 to Tier 3
    ("Nozzle Theory", "Thrust Calculation"),
    ("Combustion Chamber Design", "Thrust Calculation"),
    ("Thrust Calculation", "Atmospheric Drag"), # Drag is a counter-force to thrust
    ("Thrust Calculation", "Specific Impulse (Isp)"),

    # Tier 3 to Tier 4
    ("Specific Impulse (Isp)", "Tsiolkovsky Rocket Equation"),
    ("Rocket Staging", "Tsiolkovsky Rocket Equation"),
    ("Thrust Calculation", "Rocket Staging"),
    ("Tsiolkovsky Rocket Equation", "Orbital Mechanics")
]


# --- ENVIRONMENT PARAMETERS ---
PROFICIENCY_THRESHOLD = 0.65  # Lowered slightly to encourage progress
ASKING_COST = 0.05
MASTERY_THRESHOLD = 0.9       # Proficiency level considered "mastered"
COMPLETION_THRESHOLD = 0.95   # Percentage of topics to master to complete an episode

# --- REWARD BONUSES ---
COMPLETION_BONUS = 20.0       # Large bonus for completing the curriculum
FIRST_MASTERY_BONUS = 0.5     # Small bonus for mastering a topic for the first time


# --- AGENT HYPERPARAMETERS ---
STATE_SIZE = len(TOPICS)
ACTION_SIZE = len(TOPICS)

# PPO Specific Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ACTOR_LEARNING_RATE = 0.0003
CRITIC_LEARNING_RATE = 0.001
ENTROPY_BETA = 0.015         # Increased slightly to encourage more exploration


# --- TRAINING & FILE PARAMETERS ---
N_EPISODES = 1500            # Increased to allow for more learning time
MAX_STEPS_PER_EPISODE = 100
UPDATE_TIMESTEPS = 2048
EPOCHS_PER_UPDATE = 10

MODEL_DIR = "models"
ACTOR_MODEL_PATH = f"{MODEL_DIR}/ppo_actor.keras"
CRITIC_MODEL_PATH = f"{MODEL_DIR}/ppo_critic.keras"
GRAPH_IMAGE_PATH = "concept_graph.png"
INFERENCE_GRAPH_PATH = "inference_path_heatmap.png"

