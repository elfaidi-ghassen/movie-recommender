import tensorflow as tf
from tensorflow import keras
import numpy as np
from util.data_reader import load_ratings, inverse_dict, get_movies_pd
from util.training import cost_function, normalizeRatings, train, train_save

Y, R, movie_count, user_count, movies_id_index, users_id_index = load_ratings()
movies_index_id = inverse_dict(movies_id_index)
users_index_id = inverse_dict(users_id_index)
movies_pd = get_movies_pd()


def load_user_ratings():
    user_ratings = np.zeros(movie_count)
    user_ratings[movies_id_index[1]] = 3.5 # toy story 
    user_ratings[movies_id_index[2]] = 3.5 
    user_ratings[movies_id_index[3]] = 3.5 
    user_ratings[movies_id_index[4]] = 3.5 
    return user_ratings

def get_movie_title(movies_pd, id):
    return movies_pd.loc[movies_pd["movieId"] == id, "title"].values[0]
    

# normalize the dataset
Y_norm, Ymean = normalizeRatings(Y, R)

# TRAINING
features_count = 100
iterations = 200
lambda_ = 1

model_path = "model_params.npz"
train_save(R, Y_norm, Ymean, movie_count, user_count, features_count, iterations, lambda_, model_path)