# ============================================================
#  main.py  —  MAPPO + Cost-Aware Communication Training Loop
# ============================================================
#  history dict: added comm_actions, comm_battery keys
#  Training loop: reads env.comm_metrics() after each episode
#  Console output: shows comm actions per episode
#  Metrics plots: added comm_actions and comm_battery panels
#  Summary: includes communication efficiency stats
#  All MAPPO rollout, PPO update, visualization logic

import os

import matplotlib
_interactive = False
for _b in ("TkAgg", "Qt5Agg", "Qt6Agg", "WXAgg"):
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as _t
        _t.figure(); _t.close("all")
        _interactive = True
        break
    except Exception:
        pass
if not _interactive:
    matplotlib.use("Agg")
    print("WARNING: No interactive display — headless mode. PNG will be saved.\n")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

from environment import GridWorld
from trainer import MAPPOTrainer
from config import (
    EPISODES, STEPS_PER_EPISODE, NUM_SURVIVORS,
    GRID_SIZE, NUM_DRONES,
)


# ============================================================
#  Matplotlib figure layout  
# ============================================================
plt.style.use("seaborn-v0_8-darkgrid")
plt.ion()

fig = plt.figure(figsize=(22, 13))
fig.suptitle(
    "MAPPO + Cost-Aware Communication | Multi-Drone Search & Rescue",
    fontsize=14, fontweight="bold", y=0.985,
)

# 5 rows on right side: reward, accuracy, loss, entropy, comm
gs = gridspec.GridSpec(
    5, 2, figure=fig,
    left=0.06, right=0.97, top=0.88, bottom=0.07,
    wspace=0.38, hspace=0.75,
)

ax_grid      = fig.add_subplot(gs[:, 0])    # full-height grid panel
ax_reward    = fig.add_subplot(gs[0, 1])
ax_accuracy  = fig.add_subplot(gs[1, 1])
ax_losses    = fig.add_subplot(gs[2, 1])
ax_entropy   = fig.add_subplot(gs[3, 1])
ax_comm      = fig.add_subplot(gs[4, 1])    # communication panel


# ============================================================
#  Metric history 
# ============================================================
history = {
    "total_reward":       [],
    "accuracy":           [],
    "efficiency":         [],
    "actor_loss":         [],
    "critic_loss":        [],
    "entropy":            [],
    "success":            [],
    "comm_actions":       [],   # COMMUNICATE actions per episode
    "comm_battery":       [],   # battery spent on comm per episode
    "comm_efficiency":    [],   # rescues per comm action (or rescues if 0)
}


# ============================================================
#  Rolling average helper  
# ============================================================
def rolling(data, w=5):
    out = []
    for i in range(len(data)):
        s = max(0, i - w + 1)
        out.append(float(np.mean(data[s:i+1])))
    return out


# ============================================================
#  Grid visualisation  
# ============================================================
def draw_grid(ax, env, episode, step):
    ax.clear()
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.set_facecolor("#eef2f7")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for s in env.survivors:
        ax.add_patch(plt.Rectangle(
            (s[1] - 0.38, s[0] - 0.38), 0.76, 0.76,
            color="#e74c3c", alpha=0.85, zorder=2,
        ))

    b = env.base
    ax.scatter(b[1], b[0], s=380, marker="s",
               color="#2ecc71", edgecolors="#1a7a45",
               linewidth=2, zorder=4)

    for d in env.drones:
        ax.scatter(d.pos[1], d.pos[0], s=230,
                   color="#2980b9", edgecolors="#1a4f6e",
                   linewidth=1.5, zorder=3)

    ax.legend(
        handles=[
            mpatches.Patch(color="#2980b9", label="Drone"),
            mpatches.Patch(color="#e74c3c", label="Survivor"),
            mpatches.Patch(color="#2ecc71", label="Base"),
        ],
        loc="upper right", fontsize=8, framealpha=0.85,
    )
    found = env.survivors_found()
    comm_ep = sum(history["comm_actions"]) if history["comm_actions"] else 0
    ax.set_title(
        f"Episode {episode}/{EPISODES}  |  Step {step}\n"
        f"Rescued: {found}/{NUM_SURVIVORS}  |  Comm actions this ep: {env.comm_buffer.total_comm_actions()}",
        fontsize=10, pad=6,
    )


# ============================================================
#  Metrics panels  
# ============================================================
def draw_metrics(h, n):
    x = list(range(1, n + 1))

    def _line(ax, raw, label, color, ylim=None, title="", ylabel=""):
        ax.clear()
        ax.plot(x, raw,          color=color, alpha=0.35, linewidth=1.2)
        ax.plot(x, rolling(raw), color=color, linewidth=2, label="Rolling avg")
        ax.set_title(title,  fontsize=8, pad=3)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=6)

    _line(ax_reward,  h["total_reward"],
          "reward", "#2ecc71", title="Total Episode Reward", ylabel="Reward")

    _line(ax_accuracy, [v*100 for v in h["accuracy"]],
          "acc", "#2980b9", ylim=(0, 110),
          title="Accuracy (% Survivors Rescued)", ylabel="Acc %")

    ax_losses.clear()
    ax_losses.plot(x, h["actor_loss"],  color="#e74c3c", linewidth=1.5, label="Actor loss")
    ax_losses.plot(x, h["critic_loss"], color="#8e44ad", linewidth=1.5, label="Critic loss")
    ax_losses.plot(x, rolling(h["actor_loss"]),  color="#e74c3c", linewidth=2, linestyle="--")
    ax_losses.plot(x, rolling(h["critic_loss"]), color="#8e44ad", linewidth=2, linestyle="--")
    ax_losses.set_title("Actor & Critic Loss", fontsize=8, pad=3)
    ax_losses.set_ylabel("Loss", fontsize=7)
    ax_losses.tick_params(axis="both", labelsize=6)
    ax_losses.legend(fontsize=6)

    _line(ax_entropy, h["entropy"],
          "entropy", "#f39c12", title="Policy Entropy (Exploration)", ylabel="H(π)")

    # Communication panel — actions + battery on twin axes
    ax_comm.clear()
    ax_comm.plot(x, h["comm_actions"],         color="#1abc9c", alpha=0.35, linewidth=1.2)
    ax_comm.plot(x, rolling(h["comm_actions"]), color="#1abc9c", linewidth=2, label="Comm actions")
    ax_comm.set_title("Communication Actions & Battery Cost", fontsize=8, pad=3)
    ax_comm.set_ylabel("Comm actions", fontsize=7, color="#1abc9c")
    ax_comm.tick_params(axis="both", labelsize=6)
    ax2 = ax_comm.twinx()
    ax2.plot(x, h["comm_battery"],          color="#e67e22", alpha=0.35, linewidth=1.2)
    ax2.plot(x, rolling(h["comm_battery"]),  color="#e67e22", linewidth=2, label="Battery spent")
    ax2.set_ylabel("Battery spent", fontsize=7, color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22", labelsize=6)
    ax_comm.legend(fontsize=6, loc="upper left")
    ax2.legend(fontsize=6, loc="upper right")


# ============================================================
#  Console header  
# ============================================================
print("=" * 80)
print("  MAPPO + Cost-Aware Communication | Multi-Agent Search & Rescue")
print(f"  Drones: {NUM_DRONES}  |  Survivors: {NUM_SURVIVORS}  "
      f"|  Episodes: {EPISODES}  |  Max steps: {STEPS_PER_EPISODE}")
print("=" * 80)
print(f"{'Ep':>4}  {'Status':<10} {'Found':>5} {'Reward':>9} "
      f"{'Acc%':>6} {'Comm':>5} {'CommBat':>8} {'A-Loss':>8} {'Entropy':>8}")
print("-" * 80)


# ============================================================
#  Training loop  
# ============================================================
env     = GridWorld()
trainer = MAPPOTrainer(env.drones)

for episode in range(1, EPISODES + 1):

    global_state = env.reset()
    done = False

    # ── Collect one episode of experience ─────────────────────
    for step in range(1, STEPS_PER_EPISODE + 1):

        actions       = {}
        obs_dict      = {}
        log_prob_dict = {}

        for drone in env.drones:
            obs              = env.get_obs(drone)   # includes shared comm info
            action, log_prob = drone.select_action(obs)
            actions[drone.drone_id]       = action
            obs_dict[drone.drone_id]      = obs
            log_prob_dict[drone.drone_id] = log_prob

        rewards, done, next_global = env.step(actions)  # handles COMMUNICATE

        for drone in env.drones:
            drone.store_transition(
                obs          = obs_dict[drone.drone_id],
                action       = actions[drone.drone_id],
                log_prob     = log_prob_dict[drone.drone_id],
                reward       = rewards[drone.drone_id],
                done         = done,
                global_state = global_state,
                next_global  = next_global,
            )

        global_state = next_global

        if step % 3 == 0 or done:
            draw_grid(ax_grid, env, episode, step)
            plt.pause(0.001)

        if done:
            break

    # ── Read episode metrics BEFORE update clears buffers ─────
    ep_reward  = env.total_episode_reward()
    found      = env.survivors_found()
    accuracy   = found / NUM_SURVIVORS
    success    = 1 if found == NUM_SURVIVORS else 0
    steps_used = max(env.total_steps, 1)
    efficiency = found / steps_used

    # Read communication metrics from the environment
    cm = env.comm_metrics()
    comm_actions = cm["total_comm_actions"]
    comm_battery = cm["total_comm_battery_spent"]
    # Comm efficiency: rescues per comm action (avoid div-by-zero)
    comm_eff = found / max(comm_actions, 1)

    # ── PPO update ─────────────────────────────────────────────
    metrics = trainer.update()

    # ── Record history ─────────────────────────────────────────
    history["total_reward"].append(ep_reward)
    history["accuracy"].append(accuracy)
    history["efficiency"].append(efficiency)
    history["actor_loss"].append(metrics["actor_loss"])
    history["critic_loss"].append(metrics["critic_loss"])
    history["entropy"].append(metrics["entropy"])
    history["success"].append(success)
    history["comm_actions"].append(comm_actions)        
    history["comm_battery"].append(comm_battery)        
    history["comm_efficiency"].append(comm_eff)         

    # ── Live metric panels ─────────────────────────────────────
    draw_metrics(history, episode)
    plt.pause(0.05)

    # ── Console row   ──────────
    status = "✓ SUCCESS" if success else "✗ PARTIAL"
    print(
        f"{episode:>4}  {status:<10} {found:>2}/{NUM_SURVIVORS:<2}"
        f"  {ep_reward:>9.2f}"
        f"  {accuracy*100:>5.1f}%"
        f"  {comm_actions:>4}"
        f"  {comm_battery:>7}"
        f"  {metrics['actor_loss']:>8.4f}"
        f"  {metrics['entropy']:>8.4f}"
    )


# ============================================================
#  Training summary  
# ============================================================
print("=" * 80)
print("  TRAINING SUMMARY")
print("=" * 80)
print(f"  Mean Accuracy       : {np.mean(history['accuracy'])*100:.2f}%")
print(f"  Success Rate        : {np.mean(history['success'])*100:.2f}%")
print(f"  Mean Reward         : {np.mean(history['total_reward']):.2f}")
print(f"  Mean Efficiency     : {np.mean(history['efficiency']):.4f} survivors/step")
print(f"  Final Entropy       : {history['entropy'][-1]:.4f}")
print(f"  Final Actor Loss    : {history['actor_loss'][-1]:.4f}")
print(f"  ── Communication ──────────────────────────────────")
print(f"  Total Comm Actions  : {sum(history['comm_actions'])}")
print(f"  Avg Comm / Episode  : {np.mean(history['comm_actions']):.2f}")
print(f"  Total Comm Battery  : {sum(history['comm_battery'])}")
print(f"  Avg Comm Efficiency : {np.mean(history['comm_efficiency']):.3f} rescues/comm")
print("=" * 80)


# ============================================================
#  Final static summary plot  
# ============================================================
plt.ioff()
fig2, axes2 = plt.subplots(2, 4, figsize=(22, 9))
fig2.suptitle(
    "MAPPO + Cost-Aware Communication — Training Metrics",
    fontsize=14, fontweight="bold",
)
eps_x = list(range(1, EPISODES + 1))


def _final_plot(ax, data, title, ylabel, color, ylim=None, fill=False):
    ra = rolling(data)
    ax.plot(eps_x, data, color=color, alpha=0.3, linewidth=1.2, label="Raw")
    ax.plot(eps_x, ra,   color=color, linewidth=2.2, label="Rolling avg (5)")
    if fill:
        ax.fill_between(eps_x, ra, alpha=0.15, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)


_final_plot(axes2[0,0], history["total_reward"],
            "Total Episode Reward", "Reward", "#2ecc71", fill=True)

_final_plot(axes2[0,1], [v*100 for v in history["accuracy"]],
            "Accuracy (% Survivors Rescued)", "Accuracy %", "#2980b9",
            ylim=(0,110), fill=True)

_final_plot(axes2[0,2], history["efficiency"],
            "Efficiency (Survivors / Step)", "Surv/Step", "#8e44ad", fill=True)

# Communication actions per episode
_final_plot(axes2[0,3], history["comm_actions"],
            "Comm Actions per Episode", "# Actions", "#1abc9c", fill=True)

_final_plot(axes2[1,0], history["actor_loss"],
            "Actor Loss (PPO Clipped Surrogate)", "Loss", "#e74c3c")
axes2[1,0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

_final_plot(axes2[1,1], history["critic_loss"],
            "Critic Loss (Value MSE)", "MSE Loss", "#e67e22")

_final_plot(axes2[1,2], history["entropy"],
            "Policy Entropy (Exploration)", "H(π)", "#f39c12")

# Communication battery cost per episode
_final_plot(axes2[1,3], history["comm_battery"],
            "Battery Spent on Comm / Episode", "Battery Units", "#e74c3c", fill=True)

# Cumulative success rate overlay on accuracy
ax_s = axes2[0,1].twinx()
cumsr = [np.mean(history["success"][:i+1])*100 for i in range(EPISODES)]
ax_s.plot(eps_x, cumsr, color="#e74c3c", linewidth=1.5,
          linestyle=":", label="Cumulative success%")
ax_s.set_ylabel("Cumul. Success %", fontsize=7, color="#e74c3c")
ax_s.tick_params(axis="y", labelcolor="#e74c3c", labelsize=6)
ax_s.set_ylim(0, 110)
ax_s.legend(fontsize=7, loc="lower right")

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "mappo_comm_metrics.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved → {out}")
print("  Done.\n")
plt.show()

