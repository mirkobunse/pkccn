import numpy as np
from collections import namedtuple
from multiprocessing import Pool
from pkccn import _minimize, lima_threshold, ThresholdedClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler

class LiMaRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, p_minus, n_estimators=100, max_depth=None, n_jobs=None, random_state=None):
        self.p_minus = p_minus
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
    def fit(self, X, y_hat):
        self.classifier = ThresholdedClassifier(
            BaggingClassifier(
                LiMaTree(self.p_minus, self.max_depth),
                self.n_estimators,
                oob_score = True,
                n_jobs = self.n_jobs,
                random_state = self.random_state,
            ),
            "lima",
            prediction_method = "oob",
            method_args = {"p_minus": self.p_minus},
        ) # construct a random forest via bagging, then tune the threshold OOB
        self.classifier.fit(X, y_hat)
        self.threshold = self.classifier.threshold
        self.classes_ = self.classifier.classes_
        return self
    def predict(self, X):
        return self.classifier.predict(X)

class LiMaTree(BaseEstimator, ClassifierMixin):
    def __init__(self, p_minus, max_depth=None, random_state=None):
        self.p_minus = p_minus
        self.max_depth = max_depth
        self.random_state = random_state
    def fit(self, X, _y_hat):
        self.classes_ = np.unique(_y_hat)
        if len(self.classes_) != 2:
            raise ValueError(f"More than two classes {self.classes_}")
        y_hat = np.ones_like(_y_hat, dtype=int)
        y_hat[_y_hat==self.classes_[0]] = -1 # y_hat in [-1, +1]
        if self.random_state is None:
            rng = np.random.random.__self__ # global RNG, seeded by np.random.seed
        else:
            rng = np.random.RandomState(self.random_state) # local RNG with fixed seed
        alpha = self.p_minus / (1 - self.p_minus)
        self.tree = _construct_tree(X, y_hat, rng, alpha, self.max_depth)
        return self
    def predict_proba(self, X):
        y_pred = _predict_tree(X, self.tree)
        return np.stack((1-y_pred, y_pred)).T # P(Y=+1) to predict_proba matrix
    def predict(self, X):
        y_pred = (self.predict_proba(X)[:,1] > 0.5).astype(int)
        return self.classes_[y_pred]

_Tree = namedtuple("Tree", ["feature", "threshold", "left", "right", "y_pred"])
_Split = namedtuple("Split", ["loss", "loss_l", "loss_r", "feature", "threshold"])
def _construct_tree(X, y_hat, rng, alpha, max_depth=None, loss=0.):
    """Recursively construct a tree from noisy labels y_hat"""
    y_pred = np.sum(y_hat==1) / len(y_hat)
    if max_depth == 0 or len(np.unique(y_hat)) == 1: # leaf node with fraction of positives?
        return _Tree(None, None, None, None, y_pred)
    scaler = MinMaxScaler()
    best_split = _Split(loss, None, None, None, None) # compare with loss of parent
    for feature in rng.choice(X.shape[1], int(np.sqrt(X.shape[1]))):
        x = scaler.fit_transform(X[:,feature].reshape(-1,1)).flatten() # map to [0,1]
        t, loss, is_success = _minimize(
            _split_objective,
            10, # n_trials
            rng,
            args =(y_hat, x, alpha)
        )
        if is_success and loss < best_split.loss:
            loss_l, loss_r = _split_objective(t, y_hat, x, alpha, reduce=False)
            threshold = scaler.inverse_transform(np.array([[t]])).flatten()[0]
            best_split = _Split(loss, loss_l, loss_r, feature, threshold)
    if best_split.feature is None: # do all splits have larger loss than their parent?
        return _Tree(None, None, None, None, y_pred)
    i_left = X[:,best_split.feature] <= best_split.threshold
    i_right = np.logical_not(i_left)
    if max_depth is not None:
        max_depth -= 1
    # print(f"loss_l={-np.sqrt(-2*best_split.loss_l):.5f}, loss_r={-np.sqrt(-2*best_split.loss_r):.5f}, p={np.mean(X[:,best_split.feature] > best_split.threshold)}")
    return _Tree(
        best_split.feature,
        best_split.threshold,
        _construct_tree(X[i_left,:], y_hat[i_left], rng, alpha, max_depth, best_split.loss_l),
        _construct_tree(X[i_right,:], y_hat[i_right], rng, alpha, max_depth, best_split.loss_r),
        None, # y_pred
    )

def _split_objective(t, y_hat, x, alpha, reduce=True):
    """Objective function for lima_threshold."""
    y_left = y_hat[x <= t] # y_left is in [-1, 1]
    y_right = y_hat[x > t]
    if len(y_left) == 0 or len(y_right) == 0:
        return 0. # this threshold does not really split
    f_side = np.zeros(2) # left and right objective values
    for i_side, y_side in enumerate([y_left, y_right]):
        N = len(y_side) # N_plus + N_minus
        N_plus = np.sum(y_side == 1)
        N_minus = N - N_plus
        if N_plus < alpha * N_minus:
            continue # advance to the next side
        with np.errstate(divide='ignore', invalid='ignore'):
            f = N_plus * np.log((1+alpha)/alpha * N_plus/N) + N_minus * np.log((1+alpha) * N_minus/N)
        if not np.isfinite(f):
            continue # advance to the next side
        f_side[i_side] = -np.maximum(f, 0.) # maximize the function value
    return np.min(f_side) if reduce else f_side

def _predict_tree(X, tree):
    """Recursively predict X"""
    if tree.feature is None:
        return tree.y_pred * np.ones(len(X))
    y_pred = np.empty(len(X), dtype=float)
    i_left = X[:,tree.feature] <= tree.threshold
    i_right = np.logical_not(i_left)
    y_pred[i_left] = _predict_tree(X[i_left,:], tree.left)
    y_pred[i_right] = _predict_tree(X[i_right,:], tree.right)
    return y_pred

def _depth(tree):
    """Recursively determine the depth of a tree"""
    if tree.feature is None:
        return 0
    return max(_depth(tree.left), _depth(tree.right)) + 1
