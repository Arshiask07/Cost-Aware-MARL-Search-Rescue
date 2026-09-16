import torch.optim as optim
import matplotlib.pyplot as plt
from environment import GridWorld
from train import update_policy
from visualize import draw_grid, draw_curve
from config import EPISODES, STEPS_PER_EPISODE, NUM_SURVIVORS, GRID_SIZE

plt.style.use("seaborn-v0_8-darkgrid")
plt.ion()
fig = plt.figure(figsize=(12, 5))
ax_grid = fig.add_subplot(1, 2, 1)
ax_curve = fig.add_subplot(1, 2, 2)

survivors_found_history = []

print("\nTraining started\n")

for episode in range(EPISODES):
    env = GridWorld(optim.Adam)

    if env.done:
        print(f"Episode {episode+1}: No survivors at start, skipping.")
        survivors_found_history.append(0)
        continue

    for step in range(STEPS_PER_EPISODE):
        done = env.step()
        if step % 3 == 0:
            draw_grid(ax_grid, env, GRID_SIZE, episode, step)
            plt.pause(0.05)
        if done:
            break

    for drone in env.drones:
        update_policy(drone)

    survivors_found = NUM_SURVIVORS - len(env.survivors)
    survivors_found_history.append(survivors_found)

    print(f"Episode {episode+1} finished | Survivors found: {survivors_found}")

    draw_curve(ax_curve, survivors_found_history, NUM_SURVIVORS)
    plt.pause(0.05)

plt.ioff()
fig.savefig("results/training_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nTraining complete.")