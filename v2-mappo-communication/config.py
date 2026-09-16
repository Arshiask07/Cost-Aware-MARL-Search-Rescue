# ============================================================
#  config.py  —  Centralised Hyper-parameters & Constants
#  MAPPO: Multi-Agent Proximal Policy Optimization
#  Added cost-aware communication parameters
# ============================================================

# ── Environment ──────────────────────────────────────────────
GRID_SIZE         = 11       # NxN grid
NUM_DRONES        = 7        # number of agents
NUM_SURVIVORS     = 7        # survivors per episode
MAX_BATTERY       = 100       # steps before battery dies

# ── Training schedule ────────────────────────────────────────
EPISODES          = 150      # total training episodes
STEPS_PER_EPISODE = 80       # max env steps per episode

# ── PPO core hyper-parameters ────────────────────────────────
GAMMA             = 0.99     # discount factor
GAE_LAMBDA        = 0.95     # GAE lambda (bias-variance trade-off)
CLIP_EPSILON      = 0.2      # PPO clipping range
ENTROPY_COEF      = 0.05     # entropy bonus weight (exploration)
VALUE_LOSS_COEF   = 0.5      # critic loss weight
MAX_GRAD_NORM     = 0.5      # gradient clipping

# ── Optimiser ────────────────────────────────────────────────
LR_ACTOR          = 3e-4     # actor learning rate
LR_CRITIC         = 1e-3     # critic learning rate (usually higher)

# ── PPO update schedule ──────────────────────────────────────
PPO_EPOCHS        = 4        # gradient steps per collected batch
MINIBATCH_SIZE    = 64       # samples per minibatch

# ── Network architecture ─────────────────────────────────────
#  Actor  input  = local observation dim (see below)
#  Critic input  = global state dim (see environment.get_global_state)

# Observation now includes shared survivor positions:
#   Original 4 dims:  (row, col, battery, survivors_left_frac)
#   Added dims:       MAX_SHARED_SURVIVORS * 2  (row, col per shared survivor)
#                     + 1 (comm_used flag this step)
MAX_SHARED_SURVIVORS = 3     # max survivor positions stored in comm buffer
#                              (keeps obs fixed-size)

# ACTOR_OBS_DIM breakdown:
#   4 base dims + MAX_SHARED_SURVIVORS*2 survivor coords + 1 comm_flag
ACTOR_OBS_DIM     = 4 + MAX_SHARED_SURVIVORS * 2 + 1   # = 12

#  Global state = base(2) + survivors flat(NUM_SURVIVORS*2, zero-padded)
#                + each drone pos+battery (NUM_DRONES*3)
#  Critic still receives full global state — no change needed
GLOBAL_STATE_DIM  = 2 + NUM_SURVIVORS * 2 + NUM_DRONES * 3
ACTOR_HIDDEN      = 64
CRITIC_HIDDEN     = 128

# NUM_ACTIONS: added COMMUNICATE (action index 5)
#   0=Up, 1=Down, 2=Left, 3=Right, 4=Stay, 5=Communicate
NUM_ACTIONS       = 6

# ── Reward shaping ───────────────────────────────────────────
REWARD_SURVIVOR_FOUND  =  15.0
REWARD_ALL_FOUND_BONUS =  20.0
REWARD_STEP            =  -0.05
REWARD_BATTERY_DEAD    =  -0.5
REWARD_OUT_OF_SECTOR   =  -0.5
REWARD_PROXIMITY_SCALE =   0.3

# Communication cost parameters
COMM_BATTERY_COST    = 3      # battery units consumed per COMMUNICATE action
COMM_REWARD_PENALTY  = -0.3   # reward penalty for each communication
# Reward bonus when communication leads to a useful share (has new info)
COMM_USEFUL_BONUS    = 0.5

# ── Learning rate ────────────────────────────────────────────
LEARNING_RATE = LR_ACTOR   # kept for environment.py compatibility
