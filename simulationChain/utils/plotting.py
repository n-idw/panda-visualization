import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import transforms
from matplotlib import patches


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


def plotSTT(useGrayScale=False, alpha=1):

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
    sns.set_context("notebook")

    pipeColors = sns.color_palette("pastel")

    if useGrayScale:
        posSkewedColor = "gray"
        negSkewedColor = "darkgray"
        straightColor = "lightgray"
    else:
        posSkewedColor = pipeColors[3]
        negSkewedColor = pipeColors[0]
        straightColor = pipeColors[2]

    # Create a figure and axis
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    ax.scatter(
        posSkewedTubes["x"],
        posSkewedTubes["y"],
        ec=posSkewedColor,
        fc="none",
        s=20,
        alpha=alpha,
    )
    ax.scatter(
        negSkewedTubes["x"],
        negSkewedTubes["y"],
        ec=negSkewedColor,
        fc="none",
        s=20,
        alpha=alpha,
    )
    ax.scatter(
        straightTubes["x"],
        straightTubes["y"],
        ec=straightColor,
        fc="none",
        s=20,
        alpha=alpha,
    )

    # plotting params
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_aspect("equal")

    sns.despine(fig=fig, ax=ax, offset=10, trim=True)

    fig.tight_layout()

    return fig, ax