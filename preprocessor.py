import numpy as np
import data_handler as dta
import fitter as fit

# We keep height on the x-axis
# and mass on the y-axis.


def calculate_build_data():
    breed, height, mass = dta.get_data()
    model, params = fit.fit_curve(height, mass)

    # body_size is distance along the curve
    # build is distance above the curve
    # (breeds that fall below the curve have a negative build).
    body_size, build = convert_to_body_size(height, mass, model, params)

    return body_size, build, breed


def convert_to_body_size(height, mass, model, params):
    height_curve, mass_curve = model.to_points(params, height)
    distance, size_indices = fit.distance_to_curve(
        height, mass, height_curve, mass_curve)

    height_scale = np.var(height)
    mass_scale = np.var(mass)
    d_path = np.sqrt(
        (np.diff(height_curve)**2) / height_scale
        + (np.diff(mass_curve)**2) / mass_scale)
    path_position = np.cumsum(d_path)
    body_size = path_position[size_indices - 1]

    build = 10 * distance / np.sqrt(body_size)
    estimated_mass = model.evaluate(params, height)
    i_negative = np.where(mass < estimated_mass)[0]
    build[i_negative] = - build[i_negative]

    return body_size, build
