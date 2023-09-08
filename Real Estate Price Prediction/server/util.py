import json
import pickle
import pandas as pd

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    x = pd.DataFrame([[0] * len(__data_columns)], columns=__data_columns)
    x['total_sqft'] = sqft
    x['bath'] = bath
    x['bhk'] = bhk

    x[location] = 1

    result = round(__model.predict(x)[0], 2)

    return result if result > 0 and sqft else 0


def get_location_names():
    return __locations


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open("../models/assets/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("../models/model/banglore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Sahakara Nagar', 1000, 2, 2))  # other location
    print(get_estimated_price('Neeladri Nagar', 1000, 2, 2))  # other location
