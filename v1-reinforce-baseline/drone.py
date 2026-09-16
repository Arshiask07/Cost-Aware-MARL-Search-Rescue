import torch
from policy import PolicyNet
from config import GRID_SIZE, MAX_BATTERY

class Drone:
    def __init__(self, drone_id, start_pos, optimizer):
        self.id = drone_id
        self.sector_id = drone_id
        self.pos = start_pos
        self.battery = MAX_BATTERY

        self.policy = PolicyNet()
        self.optimizer = optimizer(self.policy.parameters())

        self.log_probs = []
        self.rewards = []

    def get_state(self, survivors_left, total_survivors):
        return torch.tensor(
            [
                self.pos[0] / GRID_SIZE,
                self.pos[1] / GRID_SIZE,
                self.battery / MAX_BATTERY,
                survivors_left / total_survivors
            ],
            dtype=torch.float32
        )

    def select_action(self, survivors_left, total_survivors):
        probs = self.policy(self.get_state(survivors_left, total_survivors))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def step(self, action):
        if self.battery <= 0:
            return -1

        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0)
        }

        dx, dy = moves[action]
        x = max(0, min(GRID_SIZE - 1, self.pos[0] + dx))
        y = max(0, min(GRID_SIZE - 1, self.pos[1] + dy))

        self.pos = (x, y)
        self.battery -= 1

        return -0.1
