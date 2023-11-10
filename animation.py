"""
Helper file for generating an environment rollout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d



def generate_animation_2d(
        episode_states,
        env,
        fps=2,
        predator_color="#C843C3",
        prey_color="#245EB6",
        runner_not_in_game_color="#666666",
        fig_width=6,
        fig_height=6,
):
    assert isinstance(episode_states, dict)

    fig, ax = plt.subplots(
        1, 1, figsize=(fig_width, fig_height)
    )  # , constrained_layout=True

    # Bounds
    ax.set_xlim(0, env.stage_size)
    ax.set_ylim(0, env.stage_size)

    # Surface
    corner_points = [(0, 0), (0, env.stage_size), (env.stage_size, env.stage_size), (env.stage_size, 0)]
    poly = Polygon(corner_points, color=runner_not_in_game_color, alpha=0.15)
    ax.add_patch(poly)

    # Hide grid lines and axes
    ax.grid(False)
    ax.set_axis_off()


    # Try to reduce whitespace
    fig.subplots_adjust(left=0, right=1, bottom=-0.2, top=1)

    # count the number of non-nan values in episode_states
    num_frames = np.count_nonzero(~np.isnan(episode_states["loc_x"][:, 0]))
    init_num_preys = env.ini_num_preys

    # Init lines
    lines = [None for _ in range(env.num_agents)]
    for idx in range(env.num_agents):
        if idx < env.ini_num_preys:  # preys
            lines[idx], = ax.plot(
                episode_states["loc_x"][:1, idx],
                episode_states["loc_y"][:1, idx],
                color=prey_color,
                marker=(3, 0, episode_states["heading"][:1, idx]),
                markersize=env.prey_radius * fig.dpi * fig_height / env.stage_size,
            )
        else:  # predators
            lines[idx], = ax.plot(
                episode_states["loc_x"][:1, idx],
                episode_states["loc_y"][:1, idx],
                color=predator_color,
                marker=(3, 0, episode_states["heading"][:1, idx]),
                markersize=env.predator_radius * fig.dpi * fig_height / env.stage_size,
            )

    labels = [None, None]
    ax.text(0, 0, "Collective Behavior\n\n", fontsize=14, color="#666666")
    labels[0] = ax.text(0, 0, "", )
    labels[1] = ax.text(0, 0, "", )
    for i, label in enumerate(labels):
        label.set_fontsize(14)
        label.set_fontweight("normal")
        label.set_color("#666666")

    # Init lines values
    def init_agent_drawing():
        labels[0].set_text("Time Step:".ljust(14) + f"{0:4.0f}\n")
        labels[1].set_text("preys Left:".ljust(14) + f"{init_num_preys:4} ({100:.0f}%)")
        return lines + labels

    # Animate
    def animate(i):
        for idx, line in enumerate(lines):

            still_in_game = episode_states["still_in_the_game"][i, idx]

            if still_in_game:
                # Update drawing
                line.set_data(
                    episode_states["loc_x"][i: i + 1, idx],
                    episode_states["loc_y"][i: i + 1, idx],
                )
                line.set_marker((3, 0, episode_states["heading"][:1, idx]))
                
            else:
                line.set_color(runner_not_in_game_color)
                line.set_marker("")

        n_preys_alive = episode_states["still_in_the_game"][i].sum() - env.num_predators
        labels[0].set_text("Time Step:".ljust(14) + f"{i:4.0f}\n")
        labels[1].set_text("preys Left:".ljust(14) + f"{n_preys_alive:4} ({n_preys_alive / init_num_preys * 100:.0f}%)")
        return lines + labels

    ani = animation.FuncAnimation(
        fig, animate, np.arange(0, num_frames), interval=1000.0 / fps, init_func=init_agent_drawing, blit=True
    )
    plt.close()

    return ani
    

def generate_animation_3d(
        episode_states,
        env,
        fps=2,
        predator_color="#C843C3",
        prey_color="#245EB6",
        runner_not_in_game_color="#666666",
        fig_width=6,
        fig_height=6,
):
    assert isinstance(episode_states, dict)

    fig, ax = plt.subplots(
        1, 1, figsize=(fig_width, fig_height)
    )  # , constrained_layout=True
    ax.remove()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Bounds
    ax.set_xlim(0, env.stage_size)
    ax.set_ylim(0, env.stage_size)
    ax.set_zlim(-0.01, 0.01)

    # Surface
    corner_points = [(0, 0), (0, env.stage_size), (env.stage_size, env.stage_size), (env.stage_size, 0)]
    poly = Polygon(corner_points, color=runner_not_in_game_color, alpha=0.15)
    ax.add_patch(poly)
    art3d.pathpatch_2d_to_3d(poly, z=0, zdir="z")

    # "Hide" side panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Hide grid lines and axes
    ax.grid(False)
    ax.set_axis_off()

    # Set camera
    ax.set_box_aspect((40, -55, 10))

    # Try to reduce whitespace
    fig.subplots_adjust(left=0, right=1, bottom=-0.2, top=1)

    # count the number of non-nan values in episode_states
    num_frames = np.count_nonzero(~np.isnan(episode_states["loc_x"][:, 0]))
    init_num_preys = env.ini_num_preys

    # Init lines
    lines = [None for _ in range(env.num_agents)]
    for idx in range(env.num_agents):
        if idx < env.ini_num_preys:  # preys
            lines[idx], = ax.plot(
                episode_states["loc_x"][:1, idx],
                episode_states["loc_y"][:1, idx],
                [0],
                color=prey_color,
                marker=(3, 0, episode_states["heading"][:1, idx]),
                markersize=env.prey_radius * fig.dpi * fig_height / env.stage_size,
            )
        else:  # predators
            lines[idx], = ax.plot(
                episode_states["loc_x"][:1, idx],
                episode_states["loc_y"][:1, idx],
                [0],
                color=predator_color,
                marker=(3, 0, episode_states["heading"][:1, idx]),
                markersize=env.predator_radius * fig.dpi * fig_height / env.stage_size,
            )

    labels = [None, None]
    ax.text(0, 0, 0.02, "Collective Behavior\n\n", fontsize=14, color="#666666")
    labels[0] = ax.text(0, 0, 0.02, "", )
    labels[1] = ax.text(0, 0, 0.02, "",
                        )
    for i, label in enumerate(labels):
        label.set_fontsize(14)
        label.set_fontweight("normal")
        label.set_color("#666666")

    # Init lines values
    def init_agent_drawing():
        labels[0].set_text("Time Step:".ljust(14) + f"{0:4.0f}\n")
        labels[1].set_text("preys Left:".ljust(14) + f"{init_num_preys:4} ({100:.0f}%)")
        return lines + labels

    # Animate
    def animate(i):
        for idx, line in enumerate(lines):

            still_in_game = episode_states["still_in_the_game"][i, idx]

            if still_in_game:
                # Update drawing
                line.set_data_3d(
                    episode_states["loc_x"][i: i + 1, idx],
                    episode_states["loc_y"][i: i + 1, idx],
                    [0],
                )
                line.set_marker((3, 0, episode_states["heading"][:1, idx]))
            else:
                line.set_color(runner_not_in_game_color)
                line.set_marker("")

        n_preys_alive = episode_states["still_in_the_game"][i].sum() - env.num_predators
        labels[0].set_text("Time Step:".ljust(14) + f"{i:4.0f}\n")
        labels[1].set_text("preys Left:".ljust(14) + f"{n_preys_alive:4} ({n_preys_alive / init_num_preys * 100:.0f}%)")
        return lines + labels

    ani = animation.FuncAnimation(
        fig, animate, np.arange(0, num_frames), interval=1000.0 / fps, init_func=init_agent_drawing, blit=True
    )
    plt.close()

    return ani
