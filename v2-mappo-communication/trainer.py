# ============================================================
#  trainer.py  —  MAPPO Training Engine
# ============================================================
#
#
#  The COMMUNICATE action is just another discrete action from the
#  PPO perspective. The actor outputs a 6-dim distribution; action=5
#  is sampled and stored like any other. The environment handles the
#  physical consequences (battery, buffer write, reward shaping).
#  The trainer sees only (obs, action, log_prob, reward, done) tuples.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict

from networks import CentralizedCritic
from drone import DroneAgent
from config import (
    GAMMA, GAE_LAMBDA, CLIP_EPSILON,
    ENTROPY_COEF, VALUE_LOSS_COEF, MAX_GRAD_NORM,
    LR_ACTOR, LR_CRITIC, PPO_EPOCHS, MINIBATCH_SIZE,
    NUM_DRONES,
)


class MAPPOTrainer:
    """
    Centralized trainer for all drone actors + one shared critic.

    Attributes:
        critic: CentralizedCritic — shared across all agents
        actor_optimizers: one Adam per drone actor
        critic_optimizer: one Adam for the critic
    """

    def __init__(self, drones: List[DroneAgent]):
        self.drones  = drones
        self.critic  = CentralizedCritic()

        self.actor_optimizers = [
            optim.Adam(drone.actor.parameters(), lr=LR_ACTOR)
            for drone in drones
        ]
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=LR_CRITIC
        )

    # ──────────────────────────────────────────────────────────
    #  GAE 
    # ──────────────────────────────────────────────────────────
    def _compute_gae(
        self,
        rewards:       torch.Tensor,
        dones:         torch.Tensor,
        global_states: torch.Tensor,
        next_globals:  torch.Tensor,
    ) -> tuple:
        T = len(rewards)

        with torch.no_grad():
            values      = self.critic(global_states).squeeze(-1)
            next_values = self.critic(next_globals).squeeze(-1)

        advantages = torch.zeros(T)
        gae        = 0.0

        for t in reversed(range(T)):
            not_done      = 1.0 - dones[t].item()
            delta         = rewards[t] + GAMMA * next_values[t] * not_done - values[t]
            gae           = delta + GAMMA * GAE_LAMBDA * not_done * gae
            advantages[t] = gae

        returns = advantages + values

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ──────────────────────────────────────────────────────────
    #  PPO update for one drone  
    # ──────────────────────────────────────────────────────────
    def _update_drone(
        self,
        drone:      DroneAgent,
        optimizer:  optim.Optimizer,
        obs:        torch.Tensor,
        actions:    torch.Tensor,
        old_lp:     torch.Tensor,
        advantages: torch.Tensor,
        returns:    torch.Tensor,
        global_st:  torch.Tensor,
    ) -> tuple:
        T = obs.shape[0]
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(T)

            for start in range(0, T, MINIBATCH_SIZE):
                mb = idx[start : start + MINIBATCH_SIZE]
                if len(mb) < 2:
                    continue

                mb_obs    = obs[mb]
                mb_acts   = actions[mb]
                mb_old_lp = old_lp[mb]
                mb_adv    = advantages[mb]
                mb_ret    = returns[mb]
                mb_gs     = global_st[mb]

                new_lp, entropy = drone.actor.evaluate_actions(mb_obs, mb_acts)
                ratio  = torch.exp(new_lp - mb_old_lp.detach())

                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * mb_adv
                a_loss = -torch.min(surr1, surr2).mean()

                values  = self.critic(mb_gs).squeeze(-1)
                c_loss  = nn.functional.mse_loss(values, mb_ret)

                total_loss = a_loss + VALUE_LOSS_COEF * c_loss - ENTROPY_COEF * entropy.mean()

                optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                if drone.drone_id == 0 and not hasattr(self, '_dbg_printed'):
                    self._dbg_printed = True
                    gnorm = sum(
                        p.grad.norm().item() ** 2
                        for p in drone.actor.parameters() if p.grad is not None
                    ) ** 0.5
                    print(f"  [DEBUG] actor grad_norm={gnorm:.6f}  ratio_mean={ratio.mean().item():.4f}  a_loss={a_loss.item():.4f}")

                nn.utils.clip_grad_norm_(drone.actor.parameters(), MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.critic.parameters(),  MAX_GRAD_NORM)

                optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(a_loss.item())
                critic_losses.append(c_loss.item())
                entropies.append(entropy.mean().item())

        mean_a = float(np.mean(actor_losses))  if actor_losses  else 0.0
        mean_c = float(np.mean(critic_losses)) if critic_losses else 0.0
        mean_e = float(np.mean(entropies))     if entropies     else 0.0
        return mean_a, mean_c, mean_e

    # ──────────────────────────────────────────────────────────
    #  Main training entry point  
    # ──────────────────────────────────────────────────────────
    def update(self) -> Dict[str, float]:
        """
        Read all drone trajectory buffers, run PPO updates,
        clear buffers, and return aggregated metrics.

        Core PPO logic.
        Communication metrics are NOT tracked here. They are
        read directly from env.comm_metrics() in main.py,
        keeping the trainer decoupled from comm details.
        """
        all_a, all_c, all_e = [], [], []

        for drone, optimizer in zip(self.drones, self.actor_optimizers):
            T = len(drone.reward_buf)
            if T < 2:
                continue

            obs        = torch.stack(drone.obs_buf)
            actions    = torch.tensor(drone.action_buf, dtype=torch.long)
            old_lp     = torch.stack(drone.log_prob_buf)
            rewards    = torch.tensor(drone.reward_buf,  dtype=torch.float32)
            dones      = torch.tensor(
                [float(d) for d in drone.done_buf], dtype=torch.float32
            )
            global_st  = torch.stack(drone.global_state_buf)
            next_gl    = torch.stack(drone.next_global_buf)

            if drone.drone_id == 0:
                print(f"\n[DEBUG] Drone 0 | Buffer T={T}")
                print(f"  rewards  : min={rewards.min():.3f}  max={rewards.max():.3f}  mean={rewards.mean():.3f}")
                print(f"  old_lp   : min={old_lp.min():.3f}  max={old_lp.max():.3f}  mean={old_lp.mean():.3f}")
                print(f"  obs[0]   : {obs[0].tolist()}")

            advantages, returns = self._compute_gae(
                rewards, dones, global_st, next_gl
            )

            if drone.drone_id == 0:
                print(f"  returns  : min={returns.min():.3f}  max={returns.max():.3f}  mean={returns.mean():.3f}")
                print(f"  advantages: min={advantages.min():.3f}  max={advantages.max():.3f}  std={advantages.std():.3f}")

            a_loss, c_loss, ent = self._update_drone(
                drone, optimizer,
                obs, actions, old_lp,
                advantages, returns, global_st,
            )

            if drone.drone_id == 0:
                print(f"  actor_loss={a_loss:.4f}  critic_loss={c_loss:.4f}  entropy={ent:.4f}")

            all_a.append(a_loss)
            all_c.append(c_loss)
            all_e.append(ent)

            drone.clear_buffers()

        return {
            "actor_loss":  float(np.mean(all_a)) if all_a else 0.0,
            "critic_loss": float(np.mean(all_c)) if all_c else 0.0,
            "entropy":     float(np.mean(all_e)) if all_e else 0.0,
        }
