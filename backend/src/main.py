import tensorflow as tf
from tensorflow import keras
import numpy as np
from util.data_reader import load_ratings, inverse_dict, get_movies_pd, get_links_pd
from util.training import normalizeRatings, train


def get_movie_title(movies_pd, id):
    return movies_pd.loc[movies_pd["movieId"] == id, "title"].values[0]

def get_movie_info(movies_pd, links_pd, id):
    title = movies_pd.loc[movies_pd["movieId"] == id, "title"].values[0]
    imdbId = links_pd.loc[links_pd["movieId"] == id, "imdbId"].values[0]
    return (title, imdbId)

# probably shouldn't load movies on every request!
def search_films(string):
    movies_pd = get_movies_pd()
    links_pd = get_links_pd()
    movies = []
    for _, movie in movies_pd.iterrows():
        if string.lower() in movie["title"].lower():
            title, imdb = get_movie_info(movies_pd, links_pd, movie["movieId"])
            movies.append((int(movie["movieId"]), title, imdb))
    return movies

def predict(movie_ids):
    recommendations = []
    Y, R, movie_count, user_count, movies_id_index, users_id_index = load_ratings()
    movies_index_id = inverse_dict(movies_id_index)
    users_index_id = inverse_dict(users_id_index)
    movies_pd = get_movies_pd()
    links_pd = get_links_pd()
    
    user_ratings = np.zeros(movie_count)
    for id in movie_ids:
        user_ratings[movies_id_index[id]] = 5

    Y = np.c_[user_ratings, Y]
    R = np.c_[(user_ratings != 0).astype(int), R]
    user_count += 1
    Y_norm, Ymean = normalizeRatings(Y, R)



    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    iterations = 5
    lambda_ = 1
    features_count = 100


    W = tf.Variable(tf.random.normal((user_count,  features_count),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((movie_count, features_count),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1,          user_count),   dtype=tf.float64),  name='b')



    X, W, b = train(X, W, b, Y_norm, R, optimizer, iterations, lambda_, debug=False)
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

    #restore the mean
    pm = p + Ymean.reshape(-1, 1)
    my_predictions = pm[:, 0]
    # sort predictions
    ix = tf.argsort(my_predictions, direction='DESCENDING')
    for i in range(30):
        j = int(ix[i].numpy())
        title, imdb = get_movie_info(movies_pd, links_pd, movies_index_id[j])
        recommendations.append((int(movies_index_id[j]), title, imdb))
    return recommendations


# movies = [5498, 1274, 134853, 26662, 5971, 31658, 26743, 2176, 2019, 6713, 109633]
# print(predict(movies))