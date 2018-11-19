import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa:E402


def plot_points(height, mass, title):
    """
    Returns the pyplot axis containing the scatterplot.
    """
    plt.figure(4390)
    plt.clf()
    ax = plt.gca()
    ax.plot(height, mass, '.')
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Dog breeds")

    return ax


def finalize_plot(ax):
    title = ax.get_title()
    title_str = "_".join(title.lower().split())
    output_filename = "mass_height_" + title_str + ".png"
    plt.savefig(output_filename)
