# Cost-Aware Search and Rescue — REINFORCE baseline

Swarm of drones searching a grid for survivors, trained with REINFORCE. Every move costs battery, so the reward has to actually push the policy toward efficient search, not just search.

Each drone has its own independent policy, no shared brain, no communication. Coordination is just a fixed sector assignment per drone, not anything learned.

## Setup

11x11 grid, 7 drones starting from a shared base at center, 7 survivors placed randomly each episode. 20 battery per drone, one unit per move (up/down/left/right/stay). Battery hits zero, drone stops.

State per drone: normalized (x, y, battery fraction, survivors-left fraction) → `Linear(4, 5)` → softmax. That's the whole policy net right now, no hidden layer.

Reward:
- -0.1 per step
- +0.3 * (distance closed to nearest survivor)
- -2 for leaving your assigned sector
- +10 for landing on a survivor
- -1 if battery's already dead

Sectors are a fixed 2x4 split of the grid, one block per drone by ID. This is what actually spreads the swarm out, not anything learned.

Update rule is standard REINFORCE: discounted return backward through the episode, `loss = -log_prob(a) * G`, one optimizer per drone, independent gradient steps.

## Known bug

`main.py` builds a new `GridWorld` inside the episode loop, which means a new `Drone` (and a freshly initialized `PolicyNet`) every episode. Policy weights don't survive between episodes, the update happens, then gets thrown away. Environment and reward logic are fine, this is purely a wiring issue in the training loop.

Also: `config.py` sets `LR = 0.01` but `drone.py` never passes it to the optimizer (`optimizer(self.policy.parameters())`, no `lr` kwarg), so it's actually training at Adam's default 0.001.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Live matplotlib grid + survivors-found curve, 40 episodes, 40 steps each, gamma 0.99. Saves to `results/training_result.png` on exit.

## Results

Survivors found bounces between 1-4 most episodes, a couple spikes to 5-6, never 7/7, no trend up or down. Given the bug above that's exactly what you'd expect 40 independent one-shot trials on a random init, not a learning curve. See `results/training_result.png`.

## If picking this back up

1. Fix the persistence bug first : move `PolicyNet`/optimizer construction outside the episode loop. Everything else downstream depends on this.
2. Give the policy a hidden layer : a bare linear layer caps what it can represent regardless.
3. Wire up `LR` properly or drop it from config.
4. Once persistence is in, a moving average on survivors-found will actually mean something.
5. Vanilla REINFORCE is high variance : a baseline/advantage term would help once there's real signal to stabilize.
6. If communication between drones is ever wanted, this is the place to add it and right now "coordination" is just hardcoded sectors.

## Files

```
main.py         training loop, live viz, entry point
environment.py  GridWorld: survivors, sectors, step/reward
drone.py        Drone: state, action selection, movement
policy.py       PolicyNet (linear, no hidden layer)
train.py        REINFORCE update
visualize.py    grid + curve rendering
config.py       hyperparams / constants
results/training_result.png
```


