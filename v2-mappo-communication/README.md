# Cost-Aware Communication in Multi-Agent Search and Rescue — MAPPO

Same task as the REINFORCE baseline, rebuilt on MAPPO with a real communication channel between drones. Fixes a persistence bug the baseline had too.

## Setup

11x11 grid, 7 drones from a shared base, 7 survivors placed randomly per episode, 100 battery per drone. Six actions now: up/down/left/right/stay/communicate.

A drone only knows about a survivor once it's physically stepped on that cell, that's tracked in its own `discovered_survivors` set and goes nowhere by default. The only way another drone finds out is the `COMMUNICATE` action, which broadcasts the sender's known positions into a shared buffer (`comm_buffer.py`). Every other drone can read from it up to 3 positions, zero-padded if there's less.

Communication isn't free:
- Costs 3 battery, same pool as movement
- Flat -0.3 reward every time
- +0.5 bonus only if the broadcast actually had new info in it and repeating what's already shared gets nothing

Rest of the reward: -0.05/step, +0.3 * distance closed, -0.5 for leaving your sector, +15 per rescue, +20 if the whole team clears the board, -0.5 if battery's dead.

## Networks

Actor per drone: local obs (12-dim: position, battery, survivors-left fraction, up to 3 shared positions, a comm-just-happened flag) → `64 → 64 → 6` → softmax. Orthogonal init.

Critic: one, shared, sees the full global state (all survivor positions, all drone positions/battery) → `128 → 128 → 1`. Training only, no drone ever sees this when picking an action.

## Training

CTDE — decentralized actors, centralized critic. Rollout a full episode, GAE for advantages (gamma 0.99, lambda 0.95), PPO clipped surrogate (clip 0.2) + entropy bonus (0.05) + value loss (0.5) + grad clipping (0.5 norm). 4 epochs per update, batch size 64. Actor LR 3e-4, critic LR 1e-3, separate optimizer per actor.

The fix from v1: drones and their actor networks get built once. `GridWorld.reset()` calls `drone.reset()` on the existing objects instead of making new ones, so weights actually carry across all 150 episodes.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

150 episodes, 80 steps max, live grid + 5 rolling metric panels. Saves `results/mappo_comm_metrics.png` at the end.

## Results

Reward has a real upward trend with rolling average starts around -190, climbs to roughly -60/-70 by the end. Noisy but real, and it's the direct payoff of fixing persistence.

Rescue rate is usually solid, 60-90% of survivors found per episode. Clearing the whole board is a different story as the cumulative full-team success rate drops off fast early and stays in the low single digits most of the run.

Comm usage settles at 70-90 actions/episode within the first ~20 episodes and just stays there and no sign of drones learning to communicate less or more selectively over time, they just converge on "communicate a lot" and hold.

Entropy drifts down from ~1.79 to ~1.63 (max for 6 actions is ln(6) ≈ 1.79) and policy gets somewhat more decisive, doesn't collapse.

Critic loss has real spikes around episodes 45, 60-65, and 100-120, up to 300-400 before it comes back down. Worth knowing about, not something the current setup fixes on its own.

Best single episode: final training episode, 7/7 rescued in 30 steps using 21 comms. See `results/result-fig 1.png` for that run and `results/mappo_comm_metrics.png` for the full 150-episode picture.

## Known issues

- `config.py` has `LEARNING_RATE = LR_ACTOR` with a comment claiming it's for `environment.py` compatibility — `environment.py` doesn't import it, it's dead. `LR_ACTOR`/`LR_CRITIC` are the ones actually wired into `trainer.py`, so this doesn't affect training, just an inaccurate comment.
- `trainer.py` has leftover `[DEBUG]` prints (drone-0 grad norms, buffer stats) firing on every update. Functional, just noisy.

## If picking this back up

- The critic loss spikes are worth digging into log survivor counts / reward magnitude / comm buffer state right before each one, see if there's a pattern.
- Comm efficiency (rescues per comm action) is already computed in `main.py`'s `history` dict but never plotted would show whether comms are getting more useful, not just staying frequent.
- Full-team clears are the biggest gap between "looks like it's learning" and "actually solves the task" and probably worth reward shaping specifically around the last 1-2 survivors.
- Clean up the dead config line and debug prints before calling this done.

## Files

```
main.py         training loop, live viz, entry point
environment.py  GridWorld: survivors, comm buffer wiring, step/reward
drone.py        DroneAgent: obs, discovery, communicate(), movement
comm_buffer.py  CommunicationBuffer: broadcast/read between drones
networks.py     Actor, CentralizedCritic
trainer.py      MAPPOTrainer: GAE + PPO update loop
config.py       hyperparams, reward values, comm costs
results/result-fig 1.png          final-episode snapshot
results/mappo_comm_metrics.png    full 150-episode metrics
```
