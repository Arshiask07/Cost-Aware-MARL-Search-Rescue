# ============================================================
#  environment.py  —  GridWorld Environment
#  MAPPO + Cost-Aware Communication version
# ============================================================
# CommunicationBuffer owned by GridWorld
# reset(): also resets comm_buffer
# get_obs(): passes shared_positions from comm_buffer
# step(): handles COMMUNICATE action separately,
# applies battery cost + reward penalty/bonus,
# records survivor discovery on cell visit
# get_global_state(), sector helpers, grid snapshot

import math
import random
import numpy as np
import torch

from drone import DroneAgent, COMMUNICATE_ACTION
from comm_buffer import CommunicationBuffer
from config import (
    GRID_SIZE, NUM_DRONES, NUM_SURVIVORS,
    REWARD_SURVIVOR_FOUND, REWARD_ALL_FOUND_BONUS,
    REWARD_BATTERY_DEAD,
    REWARD_OUT_OF_SECTOR, REWARD_PROXIMITY_SCALE,
    GLOBAL_STATE_DIM,
    COMM_REWARD_PENALTY, COMM_USEFUL_BONUS,
    MAX_SHARED_SURVIVORS,
)


class GridWorld:
    """
    NxN cooperative search-and-rescue environment.

    Interface for MAPPO + Communication:
        env.reset()                → initial global_state tensor
        env.step(actions_dict)     → rewards_dict, done, next_global_state
        env.get_obs(drone)         → local obs (includes shared comm info)
        env.get_global_state()     → full global state for critic
        env.comm_buffer            → shared CommunicationBuffer instance

    Partial observability:
        - Drones start with empty discovered_survivors sets
        - Survivors are only added to a drone's knowledge when it
          physically visits the cell (see step())
        - Shared knowledge flows only via explicit COMMUNICATE actions
    """

    def __init__(self):
        self.base            = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.done            = False
        self.total_steps     = 0
        self.total_survivors = NUM_SURVIVORS

        # ── Survivors ─────────────────────────────────────────
        self.survivors: set = set()
        while len(self.survivors) < NUM_SURVIVORS:
            pos = (
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1),
            )
            if pos != self.base:
                self.survivors.add(pos)
        self._survivor_list = list(self.survivors)

        # ── Drones ────────────────────────────────────────────
        self.drones: list = [
            DroneAgent(drone_id=i, start_pos=self.base)
            for i in range(NUM_DRONES)
        ]

        # Shared communication buffer
        self.comm_buffer = CommunicationBuffer(num_drones=NUM_DRONES)

    # ──────────────────────────────────────────────────────────
    #  Reset  
    # ──────────────────────────────────────────────────────────
    def reset(self) -> torch.Tensor:
        """
        Re-initialise episode WITHOUT replacing drone objects.

        Also resets the communication buffer so shared
        knowledge from the previous episode does not leak into the next.
        """
        self.done        = False
        self.total_steps = 0

        # Re-randomize survivors
        self.survivors = set()
        while len(self.survivors) < self.total_survivors:
            pos = (
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1),
            )
            if pos != self.base:
                self.survivors.add(pos)
        self._survivor_list = list(self.survivors)

        # Clear communication buffer for fresh episode
        self.comm_buffer.reset()

        # Reset existing drone objects in-place (preserves actor weights)
        for drone in self.drones:
            drone.reset(self.base)

        return self.get_global_state()

    # ──────────────────────────────────────────────────────────
    #  Observations  
    # ──────────────────────────────────────────────────────────
    def get_obs(self, drone: DroneAgent) -> torch.Tensor:
        """
        Local observation for a single drone.
        Shape: (ACTOR_OBS_DIM,) — now includes shared comm positions.

        Shared positions come from the comm_buffer (other drones' broadcasts).
        If no communication has occurred, shared slots are zero-padded.
        """
        # Fetch what other drones have shared
        shared_positions = self.comm_buffer.get_shared_positions(
            receiver_id   = drone.drone_id,
            max_positions = MAX_SHARED_SURVIVORS,
        )
        return drone.get_obs(
            survivors_left   = len(self.survivors),
            total_survivors  = self.total_survivors,
            shared_positions = shared_positions,
        )

    def get_global_state(self) -> torch.Tensor:
        """
        Centralized critic input.  Shape: (GLOBAL_STATE_DIM,)
        Critic always has full world visibility.
        """
        parts = []

        # Base
        parts.append(self.base[0] / (GRID_SIZE - 1))
        parts.append(self.base[1] / (GRID_SIZE - 1))

        # Survivors — use original list, mark rescued as -1
        for pos in self._survivor_list:
            if pos in self.survivors:
                parts.append(pos[0] / (GRID_SIZE - 1))
                parts.append(pos[1] / (GRID_SIZE - 1))
            else:
                parts.append(-1.0)
                parts.append(-1.0)

        # Drones
        for d in self.drones:
            parts.append(d.pos[0] / (GRID_SIZE - 1))
            parts.append(d.pos[1] / (GRID_SIZE - 1))
            parts.append(d.battery  / 50.0)

        return torch.tensor(parts, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────
    #  Sector helpers  
    # ──────────────────────────────────────────────────────────
    def _sector_bounds(self, sector_id: int) -> tuple:
        rows  = max(1, int(math.sqrt(NUM_DRONES)))
        cols  = math.ceil(NUM_DRONES / rows)
        row_h = GRID_SIZE / rows
        col_w = GRID_SIZE / cols
        r, c  = sector_id // cols, sector_id % cols
        return (
            int(r * row_h), int(min((r+1)*row_h, GRID_SIZE)),
            int(c * col_w), int(min((c+1)*col_w, GRID_SIZE)),
        )

    def _in_own_sector(self, drone: DroneAgent) -> bool:
        r0, r1, c0, c1 = self._sector_bounds(drone.sector_id)
        x, y = drone.pos
        return r0 <= x < r1 and c0 <= y < c1

    @staticmethod
    def _manhattan(p1, p2) -> int:
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    # ──────────────────────────────────────────────────────────
    #  Environment step  
    # ──────────────────────────────────────────────────────────
    def step(self, actions: dict) -> tuple:
        """
        Advance environment one timestep given a dict of actions.

        Args:
            actions: {drone_id: action_int, ...}
                     action_int == 5 → COMMUNICATE

        Returns:
            rewards (dict):         {drone_id: float}
            done    (bool):         episode over?
            global_state (Tensor):  new global state after all moves

        Communication handling:
            1. If action == COMMUNICATE:
               - drone.communicate() is called → deducts battery,
                 writes to comm_buffer
               - reward += COMM_REWARD_PENALTY  (always)
               - reward += COMM_USEFUL_BONUS    (if new info shared)
            2. After any move, if drone lands on a survivor cell:
               - drone.discover_survivor(pos) is called (partial obs)

        All sector, proximity, and rescue reward logic.
        """
        if self.done:
            return {d.drone_id: 0.0 for d in self.drones}, True, self.get_global_state()

        rewards = {}

        for drone in self.drones:
            if drone.battery <= 0:
                rewards[drone.drone_id] = REWARD_BATTERY_DEAD
                continue

            action = actions[drone.drone_id]

            # ── COMMUNICATE action ─────────────────────────────
            if action == COMMUNICATE_ACTION:
                # Base step penalty from drone.move()
                reward = drone.move(action)   # returns REWARD_STEP, no movement

                # Apply comm battery cost and buffer write
                useful = drone.communicate(self.comm_buffer)

                # Reward shaping for communication
                reward += COMM_REWARD_PENALTY
                if useful:
                    reward += COMM_USEFUL_BONUS   # bonus for sharing new info

                rewards[drone.drone_id] = reward
                self.total_steps += 1
                continue

            # ── Normal movement ────────────────────────────────
            pre_dist = (
                min(self._manhattan(drone.pos, s) for s in self.survivors)
                if self.survivors else 0
            )

            reward = drone.move(action)
            self.total_steps += 1

            # Distance shaping
            if self.survivors:
                post_dist = min(self._manhattan(drone.pos, s) for s in self.survivors)
                reward   += REWARD_PROXIMITY_SCALE * (pre_dist - post_dist)

            # Sector penalty
            if not self._in_own_sector(drone):
                reward += REWARD_OUT_OF_SECTOR

            # Survivor rescue + partial observability discovery
            if drone.pos in self.survivors:
                # Drone locally discovers this survivor
                drone.discover_survivor(drone.pos)

                self.survivors.discard(drone.pos)
                reward += REWARD_SURVIVOR_FOUND
                if len(self.survivors) == 0:
                    reward    += REWARD_ALL_FOUND_BONUS
                    self.done  = True

            rewards[drone.drone_id] = reward

        next_global = self.get_global_state()
        return rewards, self.done, next_global

    # ──────────────────────────────────────────────────────────
    #  Metrics helpers  
    # ──────────────────────────────────────────────────────────
    def survivors_found(self) -> int:
        return self.total_survivors - len(self.survivors)

    def total_episode_reward(self) -> float:
        return sum(d.episode_reward for d in self.drones)

    def comm_metrics(self) -> dict:
        """
        Return episode-level communication metrics.

        Returns dict with:
            total_comm_actions:       int — COMMUNICATE actions taken
            total_comm_battery_spent: int — battery drained by comms
        """
        return {
            "total_comm_actions":       self.comm_buffer.total_comm_actions(),
            "total_comm_battery_spent": self.comm_buffer.total_battery_spent_on_comm(),
        }

    # ──────────────────────────────────────────────────────────
    #  Grid snapshot  
    # ──────────────────────────────────────────────────────────
    def get_grid(self) -> np.ndarray:
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for s in self.survivors:
            grid[s] = 0.5
        for d in self.drones:
            grid[d.pos] = 1.0
        grid[self.base] = 0.8
        return grid
