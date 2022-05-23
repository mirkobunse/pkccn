import numpy as np

def inject_ccn(y, p_minus, p_plus, random_state=None):
    """Artificially inject CCN label noise into the clean ground-truth labels.

    :param y: an array of clean ground-truth labels in [-1, 1], shape (n,)
    :param p_minus: the negative noise rate
    :param p_plus: the positive noise rate
    :return: an array of CCN noisy labels, shape (n,)
    """
    if np.any(np.unique(y) != np.array([-1, 1])):
        raise ValueError(f"unique(y)={np.unique(y)} does not match [-1, 1]")
    rng = np.random.RandomState(random_state) # reproducible random number generator
    y_hat = np.empty_like(y)
    y_hat[y==1] = rng.choice([-1, 1], size=np.sum(y==1), p=[p_plus, 1.0 - p_plus])
    y_hat[y==-1] = rng.choice([-1, 1], size=np.sum(y==-1), p=[1.0 - p_minus, p_minus])
    return y_hat
