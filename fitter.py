import numpy as np
from scipy import optimize as opt
import data_handler as dta
import polynomial_model as mod


def fit_curve(x, y):
    model_data = dta.retrieve_model()
    if model_data:
        return model_data

    models, training_errors, testing_errors = compare_models(x, y)
    best_model = models[np.argmin(testing_errors)]
    res = train(best_model, x, y, n_iter=best_model.n_iter_default)
    p_final = res.x
    dta.save_model((best_model, p_final))

    return best_model, p_final


def compare_models(x, y, n_folds=50):
    training_errors = []
    testing_errors = []
    data_sets = []

    for i_fold in range(n_folds):
        models = mod.all_models
        data_sets.append(split_data(x, y))

    for model in models:
        all_training_errors = []
        all_testing_errors = []
        for i_fold in range(n_folds):
            x_train, y_train, x_test, y_test = data_sets[i_fold]

            res = train(model, x_train, y_train, n_iter=model.n_iter_default)
            p_final = res.x

            all_training_errors.append(loss_function(
                p_final, x_train, y_train, model))
            all_testing_errors.append(loss_function(
                p_final, x_test, y_test, model))
        training_errors.append(np.mean(all_training_errors))
        testing_errors.append(np.mean(all_testing_errors))
        print(
            "model order", model.order,
            ", training", str(training_errors[-1]),
            ", testing", str(testing_errors[-1]),
        )

    return models, training_errors, testing_errors


def split_data(x, y):
    n_train = int(y.size * .7)
    i_data = np.cumsum(np.ones(x.size), dtype=np.int) - 1
    i_train = np.sort(
        np.random.choice(i_data, size=n_train, replace=False))
    i_test = np.setdiff1d(i_data, i_train)
    x_train = x[i_train]
    x_test = x[i_test]
    y_train = y[i_train]
    y_test = y[i_test]

    return x_train, y_train, x_test, y_test


def train(model, x_train, y_train, n_iter=3):
    best_res = None
    best_loss = 1e100
    for _ in range(n_iter):
        # The arguments that will get passed to the error function,
        # in addition to the model parameters of the current iteration.
        loss_fun_args = (x_train, y_train, model)
        p_initial = model.initial_guess(x=x_train, y=y_train)
        # Confusingly the `x0` argument is not a request for the x values of
        # the data points or for the 0th element of a list. It is for the
        # initial guess for the parameter values.
        res = opt.minimize(
            fun=loss_function,
            x0=p_initial,
            method="Nelder-Mead",
            args=loss_fun_args,
        )
        loss = loss_function(
            res.x,
            x_train,
            y_train,
            model,
        )
        if loss < best_loss:
            best_loss = loss
            best_res = res

    return best_res


def loss_function(p, xs, ys, model):
    """
    The loss for any model is the average distance from all of the data points
    to the nearest point on the model curve.
    Contrary to convention, this isn't the vertical distance,
    but the minimum distance in any direction.
    """
    x_c, y_c = model.to_points(p, xs)

    distances, _ = distance_to_curve(xs, ys, x_c, y_c)
    loss = np.mean(distances)
    return loss


def distance_to_curve(x_data, y_data, x_curve, y_curve):
    x_scale = np.var(x_data)
    y_scale = np.var(y_data)
    distances = []
    indices = []
    for i, x in enumerate(x_data):
        y = y_data[i]
        deviations = np.sqrt(
            ((x_curve - x)**2) / x_scale
            + ((y_curve - y)**2) / y_scale)
        i_min = np.argmin(deviations)
        deviation = deviations[i_min]
        distances.append(deviation)
        indices.append(i_min)
    return np.array(distances), np.array(indices)
