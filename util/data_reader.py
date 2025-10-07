import pandas as pd
import numpy as np

base_path = "ml-latest-small"
ratings_data = pd.read_csv(f"{base_path}/ratings.csv")

def get_movies_pd():
    return pd.read_csv(f"{base_path}/movies.csv")
def inverse_dict(d):
    return {value:key for key,value in d.items()}

def load_ratings():
    unique_movies = ratings_data["movieId"].unique()
    unique_users = ratings_data["userId"].unique()
    movie_count = len(unique_movies)
    user_count = len(unique_users)
    
    Y = np.zeros((movie_count, user_count))
    R = np.zeros((movie_count, user_count))
    
    movies_id_index = {id:idx for idx, id in enumerate(unique_movies)}
    users_id_index = {id:idx for idx, id in enumerate(unique_users)}

    for _, row in ratings_data.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        rating = float(row["rating"])
        movie_index = movies_id_index[movie_id]
        user_index = users_id_index[user_id]
        Y[movie_index, user_index] = rating
        R[movie_index, user_index] = 1
    return Y, R, movie_count, user_count, movies_id_index, users_id_index


def test_load_ratings():
    Y, R, movies_id_index, users_id_index =  load_ratings()
    movies_index_id = inverse_dict(movies_id_index)
    users_index_id = inverse_dict(users_id_index)
    print(Y[:7, :7])
    print(f"=={Y[4]}==")
    print(movies_index_id[4], "!!")

    assert Y[movies_id_index[43558], users_id_index[448]] == 2.0
    assert R[movies_id_index[43558], users_id_index[448]] == 1
    assert R[movies_id_index[4], users_id_index[448]] == 0
    assert Y[movies_id_index[4], users_id_index[448]] == 0
    
    print("all tests pass")


if __name__ == "__main__":
    test_load_ratings()