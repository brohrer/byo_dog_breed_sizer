import data_handler as dta
import fitter as fit
import plotting as plt


breed, height, mass = dta.get_data()
scatterplot = plt.plot_points(height, mass, "Dog breeds")
plt.finalize_plot(scatterplot)

model, params = fit.fit_curve(height, mass)
height_curve, mass_curve = model.to_points(params, height)
scatterplot.plot(height_curve, mass_curve)
scatterplot.set_title("Dog breeds with " + model.name)
scatterplot.set_xlim(15, 85)
scatterplot.set_ylim(0, 100)
plt.finalize_plot(scatterplot)
