import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def draw_grid(ax, env, grid_size, episode, step):
    ax.clear()
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.set_facecolor("#f7f9fb")

    for s in env.survivors:
        ax.add_patch(
            plt.Rectangle(
                (s[1] - 0.4, s[0] - 0.4),
                0.8, 0.8,
                color="#ff6b6b",
                alpha=0.8
            )
        )

    for d in env.drones:
        ax.scatter(
            d.pos[1],
            d.pos[0],
            s=200,
            color="#1f77b4",
            edgecolors="black",
            linewidth=1.5,
            zorder=3
        )

    base = env.base
    ax.scatter(
        base[1],
        base[0],
        s=300,
        marker="s",
        color="#2ecc71",
        edgecolors="black",
        linewidth=2,
        zorder=4
    )
    ax.set_title(f"Episode {episode + 1} | Step {step + 1}")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Drone',
               markerfacecolor='#1f77b4', markeredgecolor='black', markersize=9),
        Patch(facecolor='#ff6b6b', edgecolor='none', alpha=0.8, label='Survivor'),
        Line2D([0], [0], marker='s', color='w', label='Base',
               markerfacecolor='#2ecc71', markeredgecolor='black', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0),
              fontsize=8, frameon=True, borderaxespad=0.3)


def draw_curve(ax, history, num_survivors):
    ax.clear()
    ax.plot(
        history,
        marker='o',
        color="#1f77b4",
        linewidth=2
    )
    ax.set_ylim(0, num_survivors)
    ax.set_title("Survivors Found per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Survivors Found")
    ax.grid(True)