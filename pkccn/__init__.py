import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import recall_score

class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, method, fit_classifier=True, prediction_method="training", method_args={}):
        self.base_classifier = base_classifier
        self.method = method
        self.fit_classifier = fit_classifier
        self.prediction_method = prediction_method
        self.method_args = method_args
    def fit(self, X, y_hat):
        if self.fit_classifier: # fit the base_classifier with noisy labels y_hat
            self.base_classifier.fit(X, y_hat)
        if self.prediction_method == "oob":
            y_pred = self.base_classifier.oob_decision_function_[:,1]
        elif self.prediction_method == "training":
            y_pred = self.base_classifier.predict_proba(X)[:,1]
        else:
            raise ValueError(f"prediction_method={prediction_method} is not valid")
        self.threshold = Threshold(self.method, **self.method_args)(y_hat, y_pred)
        return self
    def predict(self, X):
        return (self.base_classifier.predict_proba(X)[:,1] > self.threshold).astype(int) * 2 - 1

class Threshold:
    def __init__(self, method, **method_args):
        self.method = method
        self.method_args = method_args
    def __call__(self, y_hat, y_pred):
        if self.method == "lima":
            return lima_threshold(y_hat, y_pred, **self.method_args)
        elif self.method == "default":
            return default_threshold(y_hat, y_pred, **self.method_args)
        elif self.method == "menon":
            return menon_threshold(y_hat, y_pred, **self.method_args)
        elif self.method == "mithal":
            return mithal_threshold(y_hat, y_pred, **self.method_args)
        elif self.method == "yao":
            return yao_threshold(y_hat, y_pred, **self.method_args)
        else:
            raise ValueError(f"method=\"{self.method}\" not in [\"default\", \"menon\", \"mithal\", \"yao\"]")

def lima_score(y_true, y_threshold, p_minus):
    """Scoring function according to Li & Ma."""
    return np.sqrt(-2*__lima_objective(0, y_true, y_threshold, p_minus / (1 - p_minus)))

def f1_score(y_true, y_threshold, y_pred=None, quantiles=[.01, .99], p_minus=None, p_plus=None):
    """F1 scoring function with CCN support."""
    alpha = None
    beta = None
    pi = sum(y_true == 1) / len(y_true)
    if y_pred is not None:
        pi_corr, pi, alpha, beta, eta_min, eta_max = __menon_quantities(
            y_true, y_pred, quantiles, p_minus, p_plus
        ) # estimate all quantities from y_pred
    elif p_minus is not None and p_plus is not None:
        pi_corr, pi, alpha, beta, eta_min, eta_max = __menon_quantities(
            y_true, y_threshold, quantiles, p_minus, p_plus
        ) # define the quantities from p_minus and p_plus (ignore quantiles)
    elif p_minus is not None or p_plus is not None:
        raise ValueError(f"if y_pred is None, set both p_minus and p_plus or none of them")
    return -__f1_objective(0, y_true, y_threshold, pi, alpha, beta)

def lima_threshold(y_hat, y_pred, p_minus=None, n_trials=100, random_state=None, verbose=False):
    """Determine a clean-optimal decision threshold from noisy labels, using our proposal.

    :param y_hat: an array of noisy labels, shape (n,)
    :param y_pred: an array of soft predictions, shape (n,)
    :param p_minus: required noise rate, defaults to None
    :param n_trials: number of trials for the numerical optimization, defaults to 100
    :param random_state: optional seed for reproducibility, defaults to None
    :param verbose: whether additional information should be logged, defaults to False
    :return: a decision threshold
    """
    if p_minus is None:
        raise ValueError("p_minus is not allowed to be None")

    # optimize
    alpha = p_minus / (1 - p_minus)
    threshold, value, is_success = __minimize(
        __lima_objective,
        n_trials,
        random_state,
        args = (y_hat, y_pred, alpha)
    )

    # log and return
    if not is_success:
        print(f"WARNING: optimization in lima_threshold was not successful")
    if verbose:
        print(
            f"┌ lima_threshold={threshold}",
            f"└┬ p_minus={p_minus}",
            f" └ lima_value={np.sqrt(-2*value)}",
            sep="\n"
        )
    return threshold

def __lima_objective(threshold, y_hat, y_pred, alpha):
    """Objective function for lima_threshold."""
    y_hat = y_hat[y_pred > threshold] # y_hat is in [-1, 1]
    N = len(y_hat) # N_plus + N_minus
    N_plus = np.sum(y_hat == 1)
    N_minus = N - N_plus
    if N_plus < alpha * N_minus:
        return 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        f = N_plus * np.log((1+alpha)/alpha * N_plus/N) + N_minus * np.log((1+alpha) * N_minus/N)
    if not np.isfinite(f):
        return 0.
    return -np.maximum(f, 0.) # maximize the function value, prevent invalid sqrt

def default_threshold(y_hat, y_pred, metric="accuracy", n_trials=100, random_state=None, verbose=False):
    """Determine the default threshold for a given metric, e.g. 0.5 for accuracy.

    :param y_hat: an array of noisy labels, shape (n,)
    :param y_pred: an array of soft predictions, shape (m,)
    :param metric: the metric to optimize, defaults to "accuracy"
    :param verbose: whether additional information should be logged, defaults to False
    :return: a decision threshold
    """
    if metric == "accuracy":
        threshold = 0.5
    elif metric == "f1": # no closed-form solution; optimization is necessary
        p = np.sum(y_hat == 1) / len(y_hat) # class 1 prevalence
        threshold, value, is_success = __minimize(
            __f1_objective,
            n_trials,
            random_state,
            args = (y_hat, y_pred, p)
        )
        if not is_success:
            print(f"WARNING: f1 optimization in default_threshold was not successful")
        if verbose:
            print(
                f"┌ default_threshold={threshold}",
                f"└┬ p={p}",
                f" └ f1={-value}",
                sep="\n"
            )
        return threshold
    else:
        raise ValueError(f"metric=\"{metric}\" not in [\"accuracy\", \"f1\"]")
    if verbose:
        print(
            f"┌ default_threshold={threshold}",
            f"└─ metric={metric}",
            sep="\n"
        )
    return threshold

def __f1_objective(threshold, y_hat, y_pred, p, alpha=None, beta=None):
    """Objective function for default_threshold with metric="f1"."""
    y_pred = (y_pred > threshold).astype(int) * 2 - 1
    u = recall_score(y_hat, y_pred, pos_label=1) # u = TPR
    v = recall_score(y_hat, y_pred, pos_label=-1) # v = TNR
    if alpha is not None or beta is not None: # Sec. A.3 in [menon2015learning]
        if alpha + beta == 1:
            raise ValueError(f"Adaptation undefined for alpha={alpha} + beta={beta} == 1")
        v = 1 - ((1-alpha)*(1-v) + beta*(1-u) - beta) / (1-alpha-beta)
        u = 1 - (alpha*(1-v) + (1-beta)*(1-u) - alpha) / (1-alpha-beta)
    f = 2 * p * u / (p + (p*u + (1-p)*(1-v))) # Tab. 1 in [narasimhan2014statistical]
    return -f # maximize the function value

def menon_threshold(y_hat, y_pred, metric="accuracy", quantiles=[.01, .99], n_trials=100, random_state=None, p_minus=None, p_plus=None, verbose=False):
    """Determine a clean-optimal decision threshold from noisy labels, using the proposal by

    Menon et al. (2015): Learning from Corrupted Binary Labels via Class-Probability Estimation.

    :param y_hat: an array of noisy labels, shape (n,)
    :param y_pred: an array of soft predictions, shape (m,)
    :param metric: the metric to optimize, defaults to "accuracy"
    :param quantiles: the quantiles of y_pred, defaults to [.01, .99]
    :param verbose: whether additional information should be logged, defaults to False
    :param p_minus: optional noise rate, defaults to None
    :param p_plus: optional noise rate, defaults to None
    :return: a decision threshold
    """
    pi_corr, pi, alpha, beta, eta_min, eta_max = __menon_quantities(
        y_hat, y_pred, quantiles, p_minus, p_plus
    ) # estimate all relevant noise quantities

    # handle a corner case in which the CCN adaptation in __f1_objective is undefined
    if pi == 0: # class +1 is estimated to occur _never_
        if verbose:
            print(
                f"┌ menon_threshold={1.0}",
                f"└┬ quantiles={quantiles}",
                f" ├ metric={metric}",
                f" ├ eta_min={eta_min}, eta_max={eta_max}",
                f" ├ alpha={alpha}, beta={beta}",
                f" └ pi_corr={pi_corr}, pi={pi}",
                sep="\n"
            )
        return 1.0 # never predict class +1

    if metric == "accuracy": # compute the threshold via Eq. 12 in [menon2015learning]
        phi = lambda z : z / (1 + z)
        threshold = phi(
            pi_corr / (1 - pi_corr) *
            ((1-alpha) * (1-pi)/pi + alpha) /
            (beta * (1-pi)/pi + (1-beta))
        )
    elif metric == "f1": # no closed-form solution; optimization is necessary
        threshold, value, is_success = __minimize(
            __f1_objective,
            n_trials,
            random_state,
            args = (y_hat, y_pred, pi, alpha, beta)
        )
        if not is_success:
            print(f"WARNING: f1 optimization in menon_threshold was not successful")
    else:
        raise ValueError(f"metric=\"{metric}\" not in [\"accuracy\", \"f1\"]")

    # log and return
    if verbose:
        print(
            f"┌ menon_threshold={threshold}",
            f"└┬ quantiles={quantiles}",
            f" ├ metric={metric}",
            f" ├ eta_min={eta_min}, eta_max={eta_max}",
            f" ├ alpha={alpha}, beta={beta}",
            f" └ pi_corr={pi_corr}, pi={pi}",
            sep="\n"
        )
    return threshold

def __menon_quantities(y_hat, y_pred, quantiles=[.01, .99], p_minus=None, p_plus=None):
    """Estimate the quantities employed by Menon et al."""
    pi_corr = sum(y_hat == 1) / len(y_hat) # noisy base rate

    # estimate the noise rates via Eq. 16 / Sec. 6.3 in [menon2015learning]
    eta_min, eta_max = np.quantile(y_pred, quantiles)

    # are any probabilities known? Eq. 17 in [menon2015learning]
    if p_minus is not None:
        eta_min = p_minus
    if p_plus is not None:
        eta_max = 1.0 - p_plus

    # estimate the clean base rate, i.e. the probability of the clean-positive class
    pi = (pi_corr - eta_min) / (eta_max - eta_min) # see Sec 6.2 in [menon2015learning]

    # estimate alpha and beta
    alpha = (eta_min * (eta_max - pi_corr)) / (pi_corr * (eta_max - eta_min))
    beta = ((1 - eta_max) * (pi_corr - eta_min)) / ((1 - pi_corr) * (eta_max - eta_min))
    return (pi_corr, pi, alpha, beta, eta_min, eta_max) # return all quantities

def mithal_threshold(y_hat, y_pred, quantile=.05, n_trials=100, random_state=None, verbose=False):
    """Determine a clean-optimal decision threshold from noisy labels, using the proposal by

    Mithal et al. (2017): RAPT: Rare Class Prediction in Absence of True Labels.

    :param y_hat: an array of noisy labels, shape (n,)
    :param y_pred: an array of soft predictions, shape (n,)
    :param quantile: the quantile of y_pred, defaults to .05
    :param n_trials: number of trials for the numerical optimization, defaults to 100
    :param random_state: optional seed for reproducibility, defaults to None
    :param verbose: whether additional information should be logged, defaults to False
    :return: a decision threshold
    """
    if len(y_hat) != len(y_pred): # argument check
        raise ValueError(f"len(y_hat)={len(y_hat)} does not match len(y_pred)={len(y_pred)}")

    # estimate the beta noise rate, see page 2489 (right column middle) in [mithal2017rapt]
    is_perfectly_neg = y_pred <= np.quantile(y_pred, quantile) # perfectly negative examples = bottom 5%
    beta = np.mean(y_hat[is_perfectly_neg]) # fraction of noisy-positive samples in the bottom 5%


    # choose the threshold, see page 2489 (left column bottom) in [mithal2017rapt]
    threshold, value, is_success = __minimize(
        __mithal_objective,
        n_trials,
        random_state,
        args = (y_hat, y_pred, beta)
    )

    # log and return
    if not is_success:
        print(f"WARNING: optimization in mithal_threshold was not successful")
    if verbose:
        print(
            f"┌ mithal_threshold={threshold}",
            f"└┬ beta={beta}",
            f" └ objective_value={-value}",
            sep="\n"
        )
    return threshold

def __mithal_objective(gamma, y_hat, y_pred, beta):
    """Objective function for mithal_threshold."""
    P_g = np.mean(y_pred > gamma) # P(g(x) > gamma)
    if P_g == 0.0:
        return 0.0 # this case would otherwise result in a NaN outcome
    P_a = np.mean(y_hat[y_pred > gamma]) # P(a = 1 | g(x) > gamma)
    f = (P_a - beta)**2 * P_g
    return - f # maximize the function value

def yao_threshold(y_hat, y_pred, filter_outlier=True, verbose=False):
    """Determine a clean-optimal decision threshold from noisy labels, using the proposal by

    Yao et al. (2020): Dual T: Reducing Estimation Error for Transition Matrix in Label-noise Learning.

    :param y_hat: an array of noisy labels, shape (n,)
    :param y_pred: an array of soft predictions, shape (n,)
    :param filter_outlier: whether anchor points are found with a 97 percentile, defaults to True
    :param verbose: whether additional information should be logged, defaults to False
    :return: a decision threshold
    """
    y_proba = np.stack((1-y_pred, y_pred)).T # convert P(Y=+1) to a predict_proba matrix
    T = __yao_dual_t(y_hat, y_proba)

    # sample a grid of noisy thresholds to determine the clean threshold after applying T
    y_grid = np.arange(.000005, 1, step=.00001)
    y_grid = np.stack((1-y_grid, y_grid)).T
    y_T = np.argmax(np.matmul(y_grid, T), axis=1) # clean predictions in [0,1]
    try: # find the smallest noisy prediction for clean class +1
        threshold_plus = np.min(y_grid[y_T==1, 1])
    except ValueError:
        threshold_plus = 1.0
    try: # find the largest noisy prediction for clean class -1
        threshold_minus = np.max(y_grid[y_T==0, 1])
    except ValueError:
        threshold_minus = 0.0
    threshold = (threshold_plus + threshold_minus) / 2

    # alternative, more effective analytical implementation?
    threshold_inv = np.matmul(np.array([[.5, .5]]), np.linalg.inv(T))[0, 1]

    # log and return
    if verbose:
        print(
            f"┌ yao_threshold={threshold} -> {np.sum(y_pred>threshold) / len(y_pred)} positive",
            f"└─ threshold_inv={threshold_inv} -> {np.sum(y_pred>threshold_inv) / len(y_pred)} positive",
            sep="\n"
        )
    return threshold

def __yao_dual_t(y_hat, y_pred, filter_outlier=True):
    T_spadesuit = np.zeros((2, 2))
    pred = y_pred.argmax(axis=1)
    for i in range(len(y_hat)):
        T_spadesuit[int(pred[i])][int(y_hat[i]==1)]+=1
    T_spadesuit = np.array(T_spadesuit)
    sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(2, 1)).transpose()
    T_spadesuit = T_spadesuit/sum_matrix
    T_clubsuit = __yao_t_matrix(y_pred, filter_outlier)
    T_spadesuit = np.nan_to_num(T_spadesuit)
    return np.matmul(T_clubsuit, T_spadesuit).T

def __yao_t_matrix(y_pred, filter_outlier=True):
    T = np.empty((2, 2))
    for i in np.arange(2): # find a 'perfect example' for each class
        if not filter_outlier:
            idx_best = np.argmax(y_pred[:, i])
        else:
            eta_thresh = np.percentile(y_pred[:, i], 97, method='higher')
            robust_eta = y_pred[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(2):
            T[i, j] = y_pred[idx_best, j]
    return T

def __minimize(objective, n_trials, random_state, args=None):
    """Generic multi-start minimization of an objective function for thresholding."""
    if random_state is None:
        rng = np.random.random.__self__ # global RNG, seeded by np.random.seed
    else:
        rng = np.random.RandomState(random_state) # local RNG with fixed seed
    best = optimize.minimize_scalar(
        objective,
        bounds = (0.0, 1.0),
        method = "bounded",
        args = args
    ) # minimize_scalar is the best approach if only one local minimum exists
    for _ in range(n_trials-1):
        current = optimize.minimize(
            objective,
            rng.rand(), # random starting point
            bounds = ((0.0,1.0),),
            args = args
        ) # otherwise, we need multi-start optimization with minimize
        if current.success and (best is None or not best.success or best.fun > current.fun):
            best = current
    threshold = np.array(best.x).item() # safely convert best.x to a scalar
    return (threshold, best.fun, best.success)
