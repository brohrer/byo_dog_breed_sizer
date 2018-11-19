import pickle
import pandas as pd


data_filename = "dog_breeds.csv"
breeds_df = pd.read_csv(data_filename)


def get_data():
    breed = breeds_df["Breed"].values
    height = breeds_df["height_cm"].values
    mass = breeds_df["mass_kg"].values
    return breed, height, mass


model_filename = "height_mass_model.pkl"


def retrieve_model():
    try:
        with open(model_filename, 'rb') as model_file:
            model_data = pickle.load(model_file)
    except Exception:
        model_data = None

    return model_data


def save_model(model_data):
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model_data, model_file)
