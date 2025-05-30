import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import transforms
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Wedge
from typing import Tuple

from utils.data_processing import get_particle_tex_name


def create_rotated_cross(
    x: float, y: float, angle: float, ax: plt.axes, size=0.2, color="blue", lw=2
) -> None:
    """Function to create a rotated cross at position (x, y).

    Args:
            x (float): x coordinate of the cross
            y (float): y coordinate of the cross
            angle (float): angle of the cross in degrees
            ax (axis): axis of a given matplotlib figure
            size (float, optional): size of the cross. Defaults to 0.2.
            color (str, optional): color of the cross. Defaults to 'blue'.
            lw (int, optional): line width of the cross. Defaults to 2.
    """
    # Prepare the cross transformation (rotation and translation)
    transform = (
        transforms.Affine2D().rotate_deg(angle)
        + transforms.Affine2D().translate(x, y)
        + ax.transData
    )
    # Create a cross as a combination of two rectangles (or lines)
    cross = patches.PathPatch(
        patches.Path([(-size, 0), (size, 0), (0, 0), (0, size), (0, -size)]),
        fill=False,
        fc="None",
        ec=color,
        transform=transform,
        lw=lw,
    )
    # Add the cross to the axes
    ax.add_patch(cross)


def plot_stt(useGrayScale=False, alpha=1) -> Tuple[Figure, Axes]:

    # import the STT geometry data
    sttGeo = pd.read_csv(
        "/home/nikin105/mlProject/data/detectorGeometries/tubePos.csv",
        usecols=["x", "y", "skewed"],
    )

    posSkewedTubes = sttGeo.query("skewed == 1")
    negSkewedTubes = sttGeo.query("skewed == -1")
    straightTubes = sttGeo.query("skewed == 0")

    # Close all previous plots
    plt.close("all")

    # Set the general style of the plot using seaborn
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("talk")

    pipeColors = sns.color_palette("pastel")

    if useGrayScale:
        posSkewedColor = pipeColors[7]
        negSkewedColor = pipeColors[7]
        straightColor = pipeColors[7]
    else:
        posSkewedColor = pipeColors[3]
        negSkewedColor = pipeColors[0]
        straightColor = pipeColors[2]

    # Create a figure and axis
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # Plot the positive skewed tubes
    ax.scatter(
        posSkewedTubes["x"],
        posSkewedTubes["y"],
        ec=posSkewedColor,
        fc="none",
        s=20,
        alpha=alpha,
        lw=1,
    )
    # Plot the negative skewed tubes
    ax.scatter(
        negSkewedTubes["x"],
        negSkewedTubes["y"],
        ec=negSkewedColor,
        fc="none",
        s=20,
        alpha=alpha,
        lw=1,
    )
    # Plot the straight tubes
    ax.scatter(
        straightTubes["x"],
        straightTubes["y"],
        ec=straightColor,
        fc="none",
        s=20,
        alpha=alpha,
        lw=1,
    )

    # Format the axes
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_aspect("equal")
    sns.despine(fig=fig, ax=ax, offset=10, trim=True)

    return fig, ax


def plot_stt_on_ax(ax, useGrayScale=False, alpha=1) -> Tuple[Figure, Axes]:

    # import the STT geometry data
    sttGeo = pd.read_csv(
        "/home/nikin105/mlProject/data/detectorGeometries/tubePos.csv",
        usecols=["x", "y", "skewed"],
    )

    posSkewedTubes = sttGeo.query("skewed == 1")
    negSkewedTubes = sttGeo.query("skewed == -1")
    straightTubes = sttGeo.query("skewed == 0")

    pipeColors = sns.color_palette("pastel")

    if useGrayScale:
        posSkewedColor = pipeColors[7]
        negSkewedColor = pipeColors[7]
        straightColor = pipeColors[7]
    else:
        posSkewedColor = pipeColors[3]
        negSkewedColor = pipeColors[0]
        straightColor = pipeColors[2]

    # Plot the positive skewed tubes
    ax.scatter(
        posSkewedTubes["x"],
        posSkewedTubes["y"],
        ec=posSkewedColor,
        fc="none",
        s=20,
        alpha=alpha,
        lw=1,
    )
    # Plot the negative skewed tubes
    ax.scatter(
        negSkewedTubes["x"],
        negSkewedTubes["y"],
        ec=negSkewedColor,
        fc="none",
        s=20,
        alpha=alpha,
        lw=1,
    )
    # Plot the straight tubes
    ax.scatter(
        straightTubes["x"],
        straightTubes["y"],
        ec=straightColor,
        fc="none",
        s=20,
        alpha=alpha,
        lw=1,
    )

    # Format the axes
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_aspect("equal")

    return None


def plot_isochrones(
    x,
    y,
    isochrones,
    hid,
    color="k",
    line_width=1.0,
) -> Tuple[list, list]:

    # Prepare lists for the isochrone circles and the legend handles
    isochrone_circles = []

    for hit in hid:
        isochrone_circles.append(
            Circle(
                (x[hit], y[hit]),
                isochrones[hit],
                ec=color,
                lw=line_width,
                fc="None",
            )
        )

    return isochrone_circles


def plot_isochrones_with_pid(
    x,
    y,
    isochrones,
    n_multi_hits,
    multi_hit_pids,
    hid,
    pids,
    unique_pids,
    pdg_codes,
    pid_color_palette,
    line_width=0.15,
) -> Tuple[list, list]:

    # Prepare lists for the isochrone circles and the legend handles
    isochrone_circles = []
    legend_handles = []

    for particle_num, pid in enumerate(unique_pids):
        pid_mask = pids == pid
        particle_hid = hid[pid_mask]
        for hit in particle_hid:
            if n_multi_hits[hit] > 1:
                pids_in_hit = multi_hit_pids[hit]
                angle_per_wedge = 360 / len(pids_in_hit)
                for i, pid_in_hit in enumerate(pids_in_hit):
                    isochrone_circles.append(
                        Wedge(
                            (x[hit], y[hit]),
                            isochrones[hit].item(),
                            angle_per_wedge * i,
                            angle_per_wedge * (i + 1),
                            fc=pid_color_palette[
                                np.where(unique_pids == pid_in_hit)[0][0]
                            ],
                            width=line_width,
                            lw=0.5,
                            ec="white",
                        )
                    )
            else:
                isochrone_circles.append(
                    Wedge(
                        (x[hit], y[hit]),
                        isochrones[hit].item(),
                        0,
                        360,
                        fc=pid_color_palette[particle_num],
                        width=line_width,
                        lw=0.5,
                        ec="white",
                    )
                )
        legend_handles.append(
            plt.scatter(
                0,
                0,
                ec=pid_color_palette[particle_num],
                fc="None",
                label=f"${get_particle_tex_name(pdg_codes[pid_mask][0].item())}_{{{pid:.0f}}}$",
            )
        )
        plt.close("all")

    return isochrone_circles, legend_handles


def plot_particleIDs_and_trackIDs(
    x,
    y,
    n_multi_hits,
    multi_hit_pids,
    multi_hit_tids,
    hid,
    pids,
    tids,
    unique_pids,
    pdg_codes,
    pid_color_palette,
    tid_color_palette,
    supp_hits,
    line_width=1,
) -> Tuple[list, list]:

    # Prepare lists for the isochrone circles and the legend handles
    isochrone_circles = []
    legend_handles = []

    for particle_num, pid in enumerate(unique_pids):
        pid_mask = pids == pid
        particle_hid = hid[pid_mask]
        for hit in particle_hid:
            if hit in supp_hits:
                continue
            if n_multi_hits[hit] > 1:
                pids_in_hit = multi_hit_pids[hit]
                angle_per_wedge = 360 / len(pids_in_hit)
                for i, pid_in_hit in enumerate(pids_in_hit):
                    isochrone_circles.append(
                        Wedge(
                            (x[hit], y[hit]),
                            0.4,
                            angle_per_wedge * i,
                            angle_per_wedge * (i + 1),
                            fc=tid_color_palette[multi_hit_tids[hit][i]],
                        )
                    )
                    isochrone_circles.append(
                        Wedge(
                            (x[hit], y[hit]),
                            0.5,
                            angle_per_wedge * i,
                            angle_per_wedge * (i + 1),
                            fc=pid_color_palette[
                                np.where(unique_pids == pid_in_hit)[0][0]
                            ],
                            width=0.1,
                        )
                    )
            else:
                isochrone_circles.append(
                    Wedge(
                        (x[hit], y[hit]),
                        0.4,
                        0,
                        360,
                        fc=tid_color_palette[tids[hit]],
                        lw=line_width,
                    )
                )
                isochrone_circles.append(
                    Wedge(
                        (x[hit], y[hit]),
                        0.5,
                        0,
                        360,
                        fc=pid_color_palette[particle_num],
                        width=0.1
                    )
                )
            hit += 1
        legend_handles.append(
            plt.scatter(
                0,
                0,
                ec=pid_color_palette[particle_num],
                fc="None",
                label=f"${get_particle_tex_name(pdg_codes[pid_mask][0].item())}_{{{pid:.0f}}}$",
            )
        )
        plt.close("all")

    return isochrone_circles, legend_handles