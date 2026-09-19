<details>
<summary><b>Team Roles & Individual Contributions</b></summary>

### Arshia: Team Lead, Environment & Agent Design
* Led the project: set the overall architecture direction, made the final call on the v1/v2 split, and reviewed integration points between the environment, agent, and training code across both versions.
* Built the environment layer for both versions: `environment.py` (grid world, reward shaping, sector assignment, battery mechanics) and `drone.py` (agent state, partial observability / survivor discovery, movement).
* Designed the v2 communication mechanism (`comm_buffer.py`), including the cost structure (battery cost, penalty, new-info bonus) that makes communication a real trade-off instead of a free action.
* Proposed splitting the project into `v1-reinforce-baseline` and `v2-mappo-communication` as two clean folders instead of one branching codebase, so the baseline stayed comparable as the second version was built.

### Vasu: Algorithms & Training
* Implemented the learning side for both versions: `policy.py`/`train.py` (REINFORCE update in v1) and `networks.py`/`trainer.py` (actor/critic networks, GAE, and the PPO update loop in v2).
* Ran hyperparameter sweeps (learning rate, entropy coefficient, reward weights) across both configs.
* Caught a bug where `config.py` declares `LR = 0.01` for v1, but it's never actually passed to the optimizer in `drone.py`, so training was running at Adam's default (0.001) the whole time.

### Anjan: Evaluation, Visualization & Research
* Built the live training visualizations and metrics tracking: `visualize.py` in v1, and the multi-panel reward/accuracy/loss/entropy/comm dashboard in v2's `main.py`.
* Traced the flat, noisy v1 training curve back to its root cause: `main.py` rebuilds `GridWorld` (and every drone's `PolicyNet`) from scratch inside the episode loop, so policy weights never persist between episodes. Documented it as a known, unresolved limitation in the v1 README rather than leaving the results unexplained.
* Researched how existing multi-drone SAR systems are actually deployed and which coordination algorithms show up in the literature, and wrote the final report and presentation deck.

</details>