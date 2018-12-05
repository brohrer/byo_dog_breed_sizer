import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import data_handler as dta
import fitter as fit


def plot_points(height, mass):
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


breed, height, mass = dta.get_data()
scatterplot = plot_points(height, mass)
finalize_plot(scatterplot)

model, params = fit.fit_curve(height, mass)
height_curve, mass_curve = model.to_points(params, height)
scatterplot.plot(height_curve, mass_curve)
scatterplot.set_title("Dog breeds with " + model.name)
scatterplot.set_xlim(15, 85)
scatterplot.set_ylim(0, 100)
finalize_plot(scatterplot)
