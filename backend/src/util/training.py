import tensorflow as tf
from tensorflow import keras
import numpy as np
def cost_function(X, W, b, Y, R, lambda_):
    """
    Vectorized implementation of cost function for collaborative filtering
    X: ndarray(movies_count, features_count)
        matrix representing the features of each movie
        example:
            [
                [x1, x2, x3, ...],
                [x1, x2, x3, ...],
                [x1, x2, x3, ...], ...
            ]
        - each row describes each movie, 
        - for instance, imagine x1 represents how much action elements are in it
            and x2 how much romance in it.
    W: ndarray(user_count, features_count)
        matrix representing the parameters of each user
        example:
            [
                [w1, w2, w3, ...],
                [w1, w2, w3, ...],
                [w1, w2, w3, ...], ...
            ]
        - each row represents the "taste" of a user
        - imagine w1 is how much a user likes action movies
    b: ndarray(1, user_count)
        matrix representing the parameter b of each user
        e.g.
        [[b1, b2, b3, ...]]
        it's a single row
    Y: ndarray(movies_count, users_count)
        matrix representing the ratings of users
        e.g.
        [
            [5.0, 3.0, 1.0, 0, ...], # movie 1
            [5.0, 3.0, 0, 2.0, ...], # movie 2
        ]
        for each rating, Y[i][j] is float
        Y[i][j] in {0, 0.5, 1.0, ..., 5.0}
        0 means there is no rating
        Y[i][j] = 0 <=> R[i][j] = 0
    R: ndarray(movies_count, users_count)
        matrix that represents whether the user j rated the movie i
        R[i][j] in range {0, 1}
        if R[i][j] = 0 then the user j did not rate the movie i
        if R[i][j] = 1 then the user j rated the movie i
    lambda_: regularization parameter
    ---
    Returns:
    J: scalar tensor
        The total cost (prediction error + regularization)
    ---
    Source of the vectorized code: Andrew Ng
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J


def normalizeRatings(Y, R):
    Ymean = np.zeros((len(Y)))
    Y_norm = np.zeros_like(Y)
    for i, row in enumerate(Y):
        if sum(R[i]) == 0:
            Ymean[i] = 0
        else:
            ymean = sum(row * R[i]) / sum(R[i])
            Ymean[i] = ymean
    Y_norm = Y - Ymean.reshape(-1, 1)
    return Y_norm, Ymean


def train(X, W, b, Y, R, optimizer, iterations, lambda_, debug=False):
    for iter in range(iterations):
        with tf.GradientTape() as tape:
            cost_value = cost_function(X, W, b, Y, R, lambda_)
        grads = tape.gradient( cost_value, [X,W,b] )
        optimizer.apply_gradients( zip(grads, [X,W,b]) )
        if debug and iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
    return X, W, b


def train_save(R, Y, Ymean, movie_count, user_count, features_count, iterations, lambda_, path):
    W = tf.Variable(tf.random.normal((user_count, features_count), dtype=tf.float64), name="W")
    X = tf.Variable(tf.random.normal((movie_count, features_count),dtype=tf.float64), name="X")
    b = tf.Variable(tf.random.normal((1, user_count), dtype=tf.float64), name="b")

    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    X, W, b = train(X, W, b, Y, R, optimizer, iterations, lambda_, debug=True)
    np.savez(path, 
         W=W.numpy(), 
         X=X.numpy(), 
         b=b.numpy(), 
         Ymean=Ymean)

if __name__ == "__main__":
    Y, Ymean = normalizeRatings(np.array([[1, 2], [3, 4]]), np.array([[1, 1], [1, 1]]))
    print(Y, Ymean)