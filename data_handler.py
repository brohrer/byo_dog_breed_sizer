import os
import pickle

import pandas as pd


data_dir = "data"
try:
    os.mkdir(data_dir)
except Exception:
    pass

data_filename = "dog_breeds.csv"
data_path = os.path.sep.join([data_dir, data_filename])
breeds_df = pd.read_csv(data_path)


def get_data():
    breed = breeds_df["Breed"].values
    height = breeds_df["height_cm"].values
    mass = breeds_df["mass_kg"].values
    return breed, height, mass


model_filename = "height_mass_model.pkl"
model_path = os.path.sep.join([data_dir, model_filename])


def retrieve_model():
    try:
        with open(model_path, 'rb') as model_file:
            model_data = pickle.load(model_file)
    except Exception:
        model_data = None

    return model_data


def save_model(model_data):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model_data, model_file)


build_data_filename = "build_data.pkl"
build_data_path = os.path.sep.join([data_dir, build_data_filename])


def retrieve_build_data():
    try:
        with open(build_data_path, 'rb') as build_data_file:
            build_data = pickle.load(build_data_file)
    except Exception:
        raise
    return build_data


def save_build_data(build_data):
    with open(build_data_path, 'wb') as build_data_file:
        pickle.dump(build_data, build_data_file)
