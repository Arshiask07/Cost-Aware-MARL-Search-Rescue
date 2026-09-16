# ============================================================
#  drone.py  —  Drone Agent  (MAPPO + Communication version)
# ============================================================
#
#  ACTION_DELTAS: added action index 5 = COMMUNICATE
#  discovered_survivors: set tracking what THIS drone has found
#  communicate(): deducts battery, returns useful-flag
#  get_obs(): now includes shared comm positions + comm_flag
#                        observation is fixed-size via zero-padding
#  move(): stays in place for COMMUNICATE action
#  Actor network, trajectory buffers, select_action, PPO flow

import torch
from networks import Actor
from config import (
    GRID_SIZE, MAX_BATTERY,
    REWARD_STEP, REWARD_BATTERY_DEAD,
    ACTOR_OBS_DIM, NUM_ACTIONS,
    COMM_BATTERY_COST, MAX_SHARED_SURVIVORS,
)

# ── Action space ──────────────────────────────────────────────
# Added COMMUNICATE = 5  (no movement, costs battery)
ACTION_DELTAS = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
    4: ( 0,  0),   # Stay
    5: ( 0,  0),   # Communicate (no movement — handled separately)
}

COMMUNICATE_ACTION = 5   # sentinel constant


class DroneAgent:
    """
    A single drone agent with its own Actor network.

    Now supports:
      - Partial observability: only knows survivors IT discovered
      - COMMUNICATE action: shares known positions, costs battery
      - Extended obs vector including shared comm positions

    Trajectory buffers hold one full episode of experience.
    The MAPPOTrainer reads these buffers to perform PPO updates.
    """

    def __init__(self, drone_id: int, start_pos: tuple):
        self.drone_id  = drone_id
        self.sector_id = drone_id
        self.pos       = start_pos
        self.battery   = MAX_BATTERY

        # Partial observability: set of survivor positions this
        # drone has personally stepped on and discovered.
        self.discovered_survivors: set = set()

        # Whether this drone communicated in the CURRENT step
        # (used to set the comm_flag dimension in get_obs)
        self.comm_used_this_step: bool = False

        # Decentralized actor — now outputs NUM_ACTIONS=6
        self.actor = Actor()

        # ── Trajectory buffers ────────────────────────────────
        self.obs_buf:          list = []
        self.action_buf:       list = []
        self.log_prob_buf:     list = []
        self.reward_buf:       list = []
        self.done_buf:         list = []
        self.global_state_buf: list = []
        self.next_global_buf:  list = []

        self.episode_reward: float = 0.0

    # ──────────────────────────────────────────────────────────
    #  Observation builder
    # ──────────────────────────────────────────────────────────
    def get_obs(
        self,
        survivors_left:    int,
        total_survivors:   int,
        shared_positions:  list,   # list of (row,col) from comm buffer
    ) -> torch.Tensor:
        """
        Extended local observation including shared comm info.

        Shape: (ACTOR_OBS_DIM,)  — always fixed-size via zero-padding

        Layout:
          [0]     row  / (GRID_SIZE-1)
          [1]     col  / (GRID_SIZE-1)
          [2]     battery / MAX_BATTERY
          [3]     survivors_left / total_survivors
          [4:4+MAX_SHARED_SURVIVORS*2]
                  shared survivor positions (row,col each), zero-padded
                  if fewer than MAX_SHARED_SURVIVORS were shared
          [-1]    comm_flag: 1.0 if this drone communicated last step

         Partial observability:
          - shared_positions is populated ONLY if another drone
            communicated during a prior step in this episode.
          - By default (no comm yet) shared slots are all 0.0.
          - The comm_flag lets the actor learn to condition on
            whether fresh shared info is available.
        """
        parts = [
            self.pos[0] / (GRID_SIZE - 1),
            self.pos[1] / (GRID_SIZE - 1),
            self.battery / MAX_BATTERY,
            survivors_left / max(total_survivors, 1),
        ]

        # Append shared survivor positions (zero-padded to fixed size)
        for i in range(MAX_SHARED_SURVIVORS):
            if i < len(shared_positions):
                r, c = shared_positions[i]
                parts.append(r / (GRID_SIZE - 1))
                parts.append(c / (GRID_SIZE - 1))
            else:
                parts.append(0.0)   # padding
                parts.append(0.0)

        # Communication flag: did this drone just communicate?
        parts.append(1.0 if self.comm_used_this_step else 0.0)

        assert len(parts) == ACTOR_OBS_DIM, (
            f"Obs dim mismatch: expected {ACTOR_OBS_DIM}, got {len(parts)}"
        )
        return torch.tensor(parts, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────
    #  Action selection  (decentralized — uses only local obs)
    # ──────────────────────────────────────────────────────────
    def select_action(self, obs: torch.Tensor):
        """
        Sample action from the current actor policy.
        """
        action, log_prob = self.actor.get_action_and_log_prob(obs)
        return action, log_prob.detach()

    # ──────────────────────────────────────────────────────────
    #  Physical movement  
    # ──────────────────────────────────────────────────────────
    def move(self, action: int) -> float:
        """
        Apply action, update position & battery.
        Returns base step reward (before environment shaping).

        COMMUNICATE action (5):
          - Does NOT move the drone (dx=dy=0 already in ACTION_DELTAS)
          - Battery deduction for comm is handled in communicate()
          - Returns REWARD_STEP so the env can apply further shaping
        """
        if self.battery <= 0:
            return REWARD_BATTERY_DEAD

        if action == COMMUNICATE_ACTION:
            # Comm battery cost is applied in communicate() — do NOT
            # double-deduct here. Just return the base step penalty.
            self.comm_used_this_step = True
            return REWARD_STEP

        # Normal movement
        self.comm_used_this_step = False
        dx, dy = ACTION_DELTAS[action]
        new_x  = max(0, min(GRID_SIZE - 1, self.pos[0] + dx))
        new_y  = max(0, min(GRID_SIZE - 1, self.pos[1] + dy))
        self.pos     = (new_x, new_y)
        self.battery -= 1
        return REWARD_STEP

    # ──────────────────────────────────────────────────────────
    #  Communication  
    # ──────────────────────────────────────────────────────────
    def communicate(self, comm_buffer) -> bool:
        """
        Attempt to broadcast discovered survivor positions.

        Deducts COMM_BATTERY_COST from battery.
        Calls comm_buffer.broadcast() with this drone's known positions.

        Returns:
            True if broadcast contained NEW information (useful comm),
            False if redundant or battery dead.

        Battery check:
          We intentionally allow communicating with low battery
          (even at 1 unit) so the drone isn't silently blocked.
          If battery < COMM_BATTERY_COST the battery goes negative,
          the environment will issue REWARD_BATTERY_DEAD next step.
          This is by design — the agent must learn to conserve battery.
        """
        if self.battery <= 0:
            return False

        self.battery -= COMM_BATTERY_COST
        useful = comm_buffer.broadcast(
            sender_id    = self.drone_id,
            positions    = set(self.discovered_survivors),
            battery_cost = COMM_BATTERY_COST,
        )
        return useful

    # ──────────────────────────────────────────────────────────
    #  Survivor discovery  
    # ──────────────────────────────────────────────────────────
    def discover_survivor(self, pos: tuple):
        """
        Mark a survivor at `pos` as discovered by this drone.
        Called by the environment when the drone steps on a survivor cell.
        """
        self.discovered_survivors.add(pos)

    # ──────────────────────────────────────────────────────────
    #  Buffer management  
    # ──────────────────────────────────────────────────────────
    def store_transition(
        self,
        obs:          torch.Tensor,
        action:       int,
        log_prob:     torch.Tensor,
        reward:       float,
        done:         bool,
        global_state: torch.Tensor,
        next_global:  torch.Tensor,
    ):
        """Store one (s, a, log_π, r, done, s_global, s'_global) tuple."""
        self.obs_buf.append(obs)
        self.action_buf.append(action)
        self.log_prob_buf.append(log_prob)
        self.reward_buf.append(reward)
        self.done_buf.append(done)
        self.global_state_buf.append(global_state)
        self.next_global_buf.append(next_global)
        self.episode_reward += reward

    def clear_buffers(self):
        """Reset all trajectory buffers at the start of a new episode."""
        self.obs_buf.clear()
        self.action_buf.clear()
        self.log_prob_buf.clear()
        self.reward_buf.clear()
        self.done_buf.clear()
        self.global_state_buf.clear()
        self.next_global_buf.clear()
        self.episode_reward      = 0.0
        self.comm_used_this_step = False 

    def reset(self, start_pos: tuple):
        """Reset agent state for a new episode."""
        self.pos     = start_pos
        self.battery = MAX_BATTERY
        self.discovered_survivors.clear()  
        self.comm_used_this_step = False    
        self.clear_buffers()
