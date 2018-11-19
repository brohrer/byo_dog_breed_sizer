import numpy as np
import preprocessor as pre


def get_similar_breeds(target_breed):
    body_size, build, breed = pre.calculate_build_data()

    i_target = np.where(breed == target_breed)[0][0]
    target_size = body_size[i_target]

    build_threshold = .5
    i_lanky = np.where(build < -build_threshold)[0]
    i_stocky = np.where(build > build_threshold)[0]
    i_medium = np.where(np.logical_and(
        build >= -build_threshold,
        build <= build_threshold))[0]

    size_distance = np.abs(body_size - target_size)
    # Make sure that the target doesn't appear in the results.
    size_distance[i_target] = 1e4

    n_results = 5
    i_stocky_results = np.argsort(size_distance[i_stocky])[:n_results]
    i_medium_results = np.argsort(size_distance[i_medium])[:n_results]
    i_lanky_results = np.argsort(size_distance[i_lanky])[:n_results]
    stocky_results = breed[i_stocky][i_stocky_results]
    medium_results = breed[i_medium][i_medium_results]
    lanky_results = breed[i_lanky][i_lanky_results]

    return stocky_results, medium_results, lanky_results
