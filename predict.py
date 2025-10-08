import tensorflow as tf
from tensorflow import keras
import numpy as np
from util.data_reader import load_ratings, inverse_dict, get_movies_pd
from util.training import cost_function, normalizeRatings, train, train_save

Y, R, movie_count, user_count, movies_id_index, users_id_index = load_ratings()
movies_index_id = inverse_dict(movies_id_index)
users_index_id = inverse_dict(users_id_index)
movies_pd = get_movies_pd()
model_path = "model_params.npz"

def load_user_ratings():
    user_ratings = np.zeros(movie_count)
    user_ratings[movies_id_index[26743]] = 5 # Only Yesterday
    user_ratings[movies_id_index[31658]] = 5 # Howl's moving castle
    user_ratings[movies_id_index[5971]] = 5  # Totoro
    user_ratings[movies_id_index[26662]] = 5  # Kiki
    user_ratings[movies_id_index[134853]] = 4.5 # inside out 
    user_ratings[movies_id_index[1274]] = 4.0
    user_ratings[movies_id_index[5498]] = 5.0

    user_ratings[movies_id_index[2019]] = 4.0
    user_ratings[movies_id_index[2176]] = 5.0
    user_ratings[movies_id_index[109633]] = 5.0
    user_ratings[movies_id_index[6713]] = 5.0
    user_ratings[movies_id_index[26903]] = 5.0
    user_ratings[movies_id_index[89745]] = 1.0




    return user_ratings

def get_movie_title(movies_pd, id):
    return movies_pd.loc[movies_pd["movieId"] == id, "title"].values[0]


def load_model(path):
    data = np.load(path)
    W = tf.Variable(data["W"].astype(np.float64), dtype=tf.float64)
    X = tf.Variable(data["X"].astype(np.float64), dtype=tf.float64)
    b = tf.Variable(data["b"].astype(np.float64), dtype=tf.float64)
    Ymean = data["Ymean"].astype(np.float64)
    return W, X, b, Ymean

W, X, b, Ymean = load_model(model_path)



user_ratings = load_user_ratings()
Y = np.c_[Y, user_ratings]
R = np.c_[R, (user_ratings != 0).astype(int)]
user_idx = user_count
user_count += 1
# Y_norm = Y - Ymean.reshape(-1, 1)
Y_norm, Ymean = normalizeRatings(Y, R)

optimizer = keras.optimizers.Adam(learning_rate=0.01)
iterations = 30
lambda_ = 1
features_count = 100


new_row_np = np.zeros(features_count)
W = np.vstack([W, new_row_np])  
W = tf.Variable(W, dtype=tf.float64)

b = tf.Variable(b, dtype=tf.float64)
bk = tf.constant([[0.0]], dtype=tf.float64)
b = tf.Variable(tf.concat([b, bk], axis=1))

X = tf.Variable(X, dtype=tf.float64)
X, W, b = train(X, W, b, Y_norm, R, optimizer, iterations, lambda_, debug=True)
# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean.reshape(-1, 1)
my_predictions = pm[:,user_idx]
# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')
for i in range(1000):
    j = int(ix[i].numpy())
    print(f'Predicting rating {my_predictions[j]:0.2f} for movie {get_movie_title(movies_pd, movies_index_id[j])}')