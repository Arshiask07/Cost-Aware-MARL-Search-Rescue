import random
import math
import numpy as np
from drone import Drone
from config import GRID_SIZE, NUM_DRONES, NUM_SURVIVORS

class GridWorld:
    def __init__(self, optimizer):
        self.base = (GRID_SIZE // 2, GRID_SIZE // 2)

        # survivors
        self.survivors = set()
        while len(self.survivors) < NUM_SURVIVORS:
            pos = (
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1)
            )
            if pos != self.base:
                self.survivors.add(pos)

        self.done = (len(self.survivors) == 0)

        self.drones = [
            Drone(i, self.base, optimizer)
            for i in range(NUM_DRONES)
        ]

    # ---------- sector logic ----------
    def compute_sector_bounds(self, drone):
        n = len(self.drones)
        rows = int(math.sqrt(n))
        cols = math.ceil(n / rows)

        h = GRID_SIZE / rows
        w = GRID_SIZE / cols

        r = drone.sector_id // cols
        c = drone.sector_id % cols

        return (
            int(r * h),
            int(min((r + 1) * h, GRID_SIZE)),
            int(c * w),
            int(min((c + 1) * w, GRID_SIZE))
        )

    def in_own_sector(self, drone, x, y):
        x1, x2, y1, y2 = self.compute_sector_bounds(drone)
        return x1 <= x < x2 and y1 <= y < y2

    def manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # ---------- environment step ----------
    def step(self):
        if self.done:
            return True

        for drone in self.drones:

            if len(self.survivors) == 0:
                self.done = True
                return True

            prev_dist = min(self.manhattan(drone.pos, s) for s in self.survivors)

            action = drone.select_action(len(self.survivors), NUM_SURVIVORS)
            reward = drone.step(action)

            new_dist = min(self.manhattan(drone.pos, s) for s in self.survivors)

            # proximity shaping
            reward += 0.3 * (prev_dist - new_dist)

            # sector penalty
            if not self.in_own_sector(drone, drone.pos[0], drone.pos[1]):
                reward -= 2

            # survivor found
            if drone.pos in self.survivors:
                self.survivors.remove(drone.pos)
                reward += 10

                if len(self.survivors) == 0:
                    self.done = True

            drone.rewards.append(reward)

        return self.done

    def get_grid(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for s in self.survivors:
            grid[s] = 0.5
        for d in self.drones:
            grid[d.pos] = 1.0
        grid[self.base] = 0.8
        return grid
