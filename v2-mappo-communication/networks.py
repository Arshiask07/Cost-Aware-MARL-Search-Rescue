# ============================================================
#  networks.py  —  Actor & Centralized Critic Networks
#  Actor input dim updated for communication obs extension
# ============================================================
#
#  Actor.__init__: obs_dim now defaults to ACTOR_OBS_DIM=12
#             (was 4; now includes shared survivor positions + comm flag)
#             n_actions now defaults to NUM_ACTIONS=6 (was 5; +COMMUNICATE)
#
#  CentralizedCritic: global state dim unchanged — critic
#             still receives full world state, independent of comm obs.
#
#  CTDE paradigm :
#    - Actor: local obs (12-dim) → 6 action distribution
#    - Critic: global state (GLOBAL_STATE_DIM) → scalar V(s)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    ACTOR_OBS_DIM, GLOBAL_STATE_DIM,
    ACTOR_HIDDEN, CRITIC_HIDDEN, NUM_ACTIONS
)


class Actor(nn.Module):
    """
    Per-agent stochastic policy network.

    Architecture:
        obs (12,) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(6) → Softmax

        obs dim 12 = 4 base + MAX_SHARED_SURVIVORS*2 shared positions + 1 comm flag
        action dim 6 = Up, Down, Left, Right, Stay, Communicate

    Weight init, forward(), evaluate_actions() logic.
    """

    def __init__(
        self,
        obs_dim:    int = ACTOR_OBS_DIM,    
        hidden_dim: int = ACTOR_HIDDEN,
        n_actions:  int = NUM_ACTIONS,      
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for policy gradient networks.
        Hidden layers: gain=sqrt(2). Output layer: gain=0.01.
        """
        layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for layer in layers[:-1]:
            nn.init.orthogonal_(layer.weight, gain=1.4142)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(layers[-1].weight, gain=0.01)
        nn.init.zeros_(layers[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """
        Forward pass: return a Categorical distribution over 6 actions.
        """
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def get_action_and_log_prob(self, obs: torch.Tensor):
        """Sample action + return its log-probability. Used during rollout."""
        dist   = self.forward(obs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Re-evaluate stored actions under the current (updated) policy.
        Returns log_probs and entropy for the PPO loss.
        
        """
        dist      = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, entropy


class CentralizedCritic(nn.Module):
    """
    Shared centralized value network.

    The critic receives the full global state which already
    contains complete world information. Adding communication does NOT
    change the global state encoding. The critic's superior information
    access makes comm-info redundant at its level.

    Architecture:
        global_state (GLOBAL_STATE_DIM,)
            → Linear(128) → ReLU → Linear(128) → ReLU → Linear(1)
    """

    def __init__(
        self,
        state_dim:  int = GLOBAL_STATE_DIM,
        hidden_dim: int = CRITIC_HIDDEN,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: (batch, GLOBAL_STATE_DIM) or (GLOBAL_STATE_DIM,)
        Returns:
            value: (batch, 1) or (1,)
        """
        return self.net(global_state)
