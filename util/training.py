import tensorflow as tf

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