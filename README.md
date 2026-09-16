# Cost-Aware Multi-Agent Search and Rescue

Two versions of the same problem: a swarm of drones searching a grid for survivors under a battery budget. `v1` is a REINFORCE baseline. `v2` is a MAPPO rebuild that adds real inter-agent communication and fixes a persistence bug that was quietly killing v1's training.

## TL;DR

| | v1 — REINFORCE | v2 — MAPPO + Comm |
|---|---|---|
| Algorithm | REINFORCE (Monte Carlo policy gradient) | MAPPO (PPO, centralized critic) |
| Communication | None | `COMMUNICATE` action, shared buffer, costs battery + reward |
| Policy persistence | **Broken** — new `GridWorld`/policy every episode | Fixed — drones persist across all 150 episodes |
| Policy net | `Linear(4, 5)`, no hidden layer | MLP, 2×64 hidden |
| Value function | None | Centralized critic, 2×128 hidden |
| Action space | 5 | 6 (+Communicate) |
| Episodes | 40 | 150 |
| Battery | 20 | 100 |
| Result | Flat, noisy — 1-4 survivors/ep, no trend | Reward climbs from ~-190 to ~-60/-70; 60-90% rescue rate common; 7/7 still rare |

## v1 — REINFORCE baseline

Each drone runs its own independent policy, no communication, no value function and just discounted return driving a REINFORCE update at the end of each episode. Environment and reward shaping both work fine. The problem is in `main.py`: `GridWorld` gets constructed *inside* the episode loop, which means every drone's `PolicyNet` gets reinitialized from scratch every episode. There's a second, smaller bug too, `config.py` sets `LR = 0.01` but it's never actually passed to the optimizer, so every drone trains at Adam's default LR (0.001) instead.

Net effect: 40 episodes that each look like an independent one-shot trial rather than a run that's actually learning anything. Survivors found bounces around 1-4 with a couple of spikes to 5-6, no upward trend. That's expected given the bug, not a REINFORCE failure. Full writeup and the actual training curve are in [`v1-reinforce-baseline/README.md`](v1-reinforce-baseline/README.md).

## v2 — MAPPO + cost-aware communication

Fixes the persistence bug (drones and their actor networks now live outside the episode loop and just get `reset()`, not recreated) and adds a real 6th action: `COMMUNICATE`. A drone only knows about a survivor once it's physically stepped on that cell. The only way that knowledge reaches a teammate is a broadcast into a shared buffer, and broadcasting costs 3 battery and a flat -0.3 reward hit, with a +0.5 bonus only if the broadcast has new info in it.

Trained for 150 episodes with a centralized critic + PPO. Reward has a real upward trend this time (~-190 early to ~-60/-70 late), rescue rate is usually solid (60-90% of survivors found per episode), and the best run cleared all 7 in 30 steps using 21 comms. Full board clears stay rare, and critic loss has some real instability spikes mid-training. Full metrics, plots, and known issues (a dead config line, some leftover debug prints) are in [`v2-mappo-communication/README.md`](v2-mappo-communication/README.md).

## Repo layout

```
.
├── v1-reinforce-baseline/
│   ├── main.py, drone.py, environment.py, policy.py, train.py, visualize.py, config.py
│   ├── results/training_result.png
│   └── README.md
├── v2-mappo-communication/
│   ├── main.py, drone.py, environment.py, comm_buffer.py, networks.py, trainer.py, config.py
│   ├── results/result-fig 1.png, results/mappo_comm_metrics.png
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
cd v1-reinforce-baseline   # or v2-mappo-communication
python main.py
```

Opens a live matplotlib window during training, saves results to that version's own `results/` folder when it's done.
