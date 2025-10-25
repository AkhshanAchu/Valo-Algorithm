import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_simulation(positions_history, best_history, FUNC, FUNC_NAME, LB, UB, roles, N_AGENTS, ITER, output_gif=True):
    # prepare contour grid
    n_grid = 200
    x = np.linspace(LB[0], UB[0], n_grid)
    y = np.linspace(LB[1], UB[1], n_grid)
    Xg, Yg = np.meshgrid(x, y)
    Z = np.vectorize(lambda a, b: FUNC([a, b]))(Xg, Yg)

    # role colors
    role_colors = {"initiator": "blue", "duelist": "red", "controller": "green", "sentinel": "purple"}

    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contourf(Xg, Yg, Z, levels=60, cmap='viridis', alpha=0.75)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('Fitness')

    ax.set_title(f'LISTEN Simulation on {FUNC_NAME}')
    ax.set_xlim(LB[0], UB[0])
    ax.set_ylim(LB[1], UB[1])

    # scatter objects for agents
    scatters = []
    for role in ["initiator", "duelist", "controller", "sentinel"]:
        sc = ax.scatter([], [], s=60, label=f"{role}", edgecolors='k', linewidths=0.4)
        scatters.append(sc)

    # best marker
    best_marker, = ax.plot([], [], marker='*', markersize=18, color='yellow',
                           markeredgecolor='k', markeredgewidth=0.8, linestyle='None', label='Best')

    # trails
    trail_lines = [ax.plot([], [], lw=1.2, alpha=0.9)[0] for _ in range(6)]

    ax.legend(loc='upper right')

    # init animation
    def init():
        for sc in scatters:
            sc.set_offsets(np.empty((0, 2)))
        best_marker.set_data([], [])
        for tr in trail_lines:
            tr.set_data([], [])
        return scatters + [best_marker] + trail_lines

    # update animation
    def update(frame):
        pos = positions_history[frame]
        for idx, role in enumerate(["initiator", "duelist", "controller", "sentinel"]):
            inds = [i for i, r in enumerate(roles) if r == role]
            if len(inds) > 0:
                coords = pos[inds]
                scatters[idx].set_offsets(coords)
                scatters[idx].set_facecolor(role_colors[role])
                scatters[idx].set_edgecolor('k')
                scatters[idx].set_sizes([80 if role == 'duelist' else 60 for _ in coords])

        best_pos = best_history[frame]
        if best_pos is not None:
            best_marker.set_data([best_pos[0]], [best_pos[1]])

        for j in range(min(6, N_AGENTS)):
            agent_traj = np.array([positions_history[t][j] for t in range(max(0, frame-30), frame+1)])
            if len(agent_traj) > 0:
                trail_lines[j].set_data(agent_traj[:, 0], agent_traj[:, 1])
                trail_lines[j].set_alpha(0.9)
            else:
                trail_lines[j].set_data([], [])

        ax.set_xlabel(f"Iteration: {frame}/{ITER}")
        return scatters + [best_marker] + trail_lines

    anim = animation.FuncAnimation(fig, update, frames=len(positions_history), init_func=init,
                                   blit=False, interval=120, repeat=False)

    if output_gif:
        out_path = f"listen_simulation_{FUNC_NAME}.gif"
        try:
            writergif = animation.PillowWriter(fps=8)
            anim.save(out_path, writer=writergif)
            print("Saved animation to", out_path)
        except Exception as e:
            print("Could not save GIF:", e)

    plt.show()
