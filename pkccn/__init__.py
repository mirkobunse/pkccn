import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, method, fit_classifier=True, method_args={}):
        self.base_classifier = base_classifier
        self.method = method
        self.fit_classifier = fit_classifier
        self.method_args = method_args
    def fit(self, X, y_hat):
        if self.fit_classifier: # fit the base_classifier with noisy labels y_hat
            self.base_classifier.fit(X, y_hat)
        y_pred = self.base_classifier.predict_proba(X)[:,1]
        if self.method == "none":
            self.threshold = 0.5
        elif self.method == "menon":
            self.threshold = menon_threshold(y_hat, y_pred, **self.method_args)
        elif self.method == "mithal":
            self.threshold = mithal_threshold(y_hat, y_pred, **self.method_args)
        else:
            raise ValueError(f"method=\"{self.method}\" not in [\"none\", \"menon\", \"mithal\"]")
        return self
    def predict(self, X):
        return (self.base_classifier.predict_proba(X)[:,1] > self.threshold).astype(int) * 2 - 1

def menon_threshold(y_hat, y_pred, metric="accuracy", quantiles=[.01, .99], verbose=False, p_minus=None, p_plus=None):
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
    pi_corr = sum(y_hat == 1) / len(y_hat) # noisy base rate

    # estimate the noise rates via Eq. 16 / Sec. 6.3 in [menon2015learning]
    eta_min, eta_max = np.quantile(y_pred, quantiles)

    # are any probabilities known? Eq. 17 in [menon2015learning]
    if p_minus is not None:
        eta_min = p_minus
    if p_plus is not None:
        eta_max = 1.0 - p_plus

    alpha = (eta_min * (eta_max - pi_corr)) / (pi_corr * (eta_max - eta_min))
    beta = ((1 - eta_max) * (pi_corr - eta_min)) / ((1 - pi_corr) * (eta_max - eta_min))

    # estimate the clean base rate, i.e. the probability of the clean-positive class
    pi = (pi_corr - eta_min) / (eta_max - eta_min) # see Sec 6.2 in [menon2015learning]

    if metric == "accuracy": # compute the threshold via Eq. 12 in [menon2015learning]
        phi = lambda z : z / (1 + z)
        threshold = phi(
            pi_corr / (1 - pi_corr) *
            ((1-alpha) * (1-pi)/pi + alpha) /
            (beta * (1-pi)/pi + (1-beta))
        )
    elif metric == "f1":
        raise ValueError("f1 not yet implemented") # TODO
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

def mithal_threshold(y_hat, y_pred, quantile=.05, verbose=False, n_trials=100):
    """Determine a clean-optimal decision threshold from noisy labels, using the proposal by

    Mithal et al. (2017): RAPT: Rare Class Prediction in Absence of True Labels.

    :param y_hat: an array of noisy labels, shape (n,)
    :param y_pred: an array of soft predictions, shape (n,)
    :param quantile: the quantile of y_pred, defaults to .05
    :param verbose: whether additional information should be logged, defaults to False
    :param n_trials: number of trials for the numerical optimization, defaults to 100
    :return: a decision threshold
    """
    if len(y_hat) != len(y_pred): # argument check
        raise ValueError(f"len(y_hat)={len(y_hat)} does not match len(y_pred)={len(y_pred)}")

    # estimate the beta noise rate, see page 2489 (right column middle) in [mithal2017rapt]
    is_perfectly_neg = y_pred <= np.quantile(y_pred, quantile) # perfectly negative examples = bottom 5%
    beta = np.mean(y_hat[is_perfectly_neg]) # fraction of noisy-positive samples in the bottom 5%

    # choose the threshold, see page 2489 (left column bottom) in [mithal2017rapt]
    def objective(gamma):
        P_g = np.mean(y_pred > gamma) # P(g(x) > gamma)
        if P_g == 0.0:
            return 0.0 # this case would otherwise result in a NaN outcome
        P_a = np.mean(y_hat[y_pred > gamma]) # P(a = 1 | g(x) > gamma)
        obj = (P_a - beta)**2 * P_g
        return - obj # maximize obj
    best_res = optimize.minimize_scalar(objective, method="Bounded", bounds=(0., 1.))
    for _ in range(n_trials-1):
        res = optimize.minimize(objective, np.random.rand(), bounds=((0.0,1.0),))
        if res.success and (best_res is None or not best_res.success or best_res.fun > res.fun):
            best_res = res
    threshold = np.array(best_res.x).item() # safely convert best_res.x to a scalar

    # log and return
    if not best_res.success:
        print(f"WARNING: optimization in mithal_threshold was not successful")
    if verbose:
        print(
            f"┌ mithal_threshold={threshold}",
            f"└┬ beta={beta}",
            f" └ fun={-best_res.fun}, nfev={best_res.nfev}",
            sep="\n"
        )
    return threshold
