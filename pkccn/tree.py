import numpy as np
from collections import namedtuple
from multiprocessing import Pool
from pkccn import lima_threshold, ThresholdedClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler

class LiMaRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, p_minus, n_estimators=100, max_depth=None, n_jobs=None):
        self.p_minus = p_minus
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
    def fit(self, X, y_hat):
        self.classifier = ThresholdedClassifier(
            BaggingClassifier(
                LiMaTree(self.p_minus, self.max_depth),
                self.n_estimators,
                oob_score = True,
                n_jobs = self.n_jobs,
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
    def __init__(self, p_minus, max_depth=None):
        self.p_minus = p_minus
        self.max_depth = max_depth
    def fit(self, X, _y_hat):
        self.classes_ = np.unique(_y_hat)
        if len(self.classes_) != 2:
            raise ValueError(f"More than two classes {self.classes_}")
        y_hat = np.ones_like(_y_hat, dtype=int)
        y_hat[_y_hat==self.classes_[0]] = -1 # y_hat in [-1, +1]
        self.tree = _construct_tree(X, y_hat, self.p_minus, self.max_depth)
        return self
    def predict_proba(self, X):
        y_pred = _predict_tree(X, self.tree)
        return np.stack((1-y_pred, y_pred)).T # P(Y=+1) to predict_proba matrix
    def predict(self, X):
        y_pred = (self.predict_proba(X)[:,1] > 0.5).astype(int)
        return self.classes_[y_pred]

__Tree = namedtuple("Tree", ["feature", "threshold", "left", "right", "y_pred"])
def _construct_tree(X, y_hat, p_minus, remaining_depth=None):
    """Recursively construct a tree from noisy labels y_hat"""
    if remaining_depth == 0 or len(X) == 1: # leaf node with fraction of positives?
        return __Tree(None, None, None, None, np.sum(y_hat==1))
    scaler = MinMaxScaler()
    best_split = (0, None, None) # (score, feature, threshold)
    for feature in np.random.choice(X.shape[1], int(np.sqrt(X.shape[1]))):
        x = scaler.fit_transform(X[:,feature].reshape(-1,1)).flatten() # map to [0,1]
        t, score = lima_threshold(y_hat, x, p_minus, return_score=True, n_trials=10)
        if score > best_split[0]:
            best_split = (score, feature, scaler.inverse_transform(np.array([[t]]))[0])
    if best_split[1] is None: # do all splits have a score of 0?
        return __Tree(None, None, None, None, np.sum(y_hat==1))
    i_left = X[:,best_split[1]] < best_split[2] # X[:, feature] < threshold
    i_right = np.logical_not(i_left)
    if remaining_depth is not None:
        remaining_depth -= 1
    return __Tree(
        best_split[1], # feature
        best_split[2], # threshold
        _construct_tree(X[i_left,:], y_hat[i_left], p_minus, remaining_depth),
        _construct_tree(X[i_right,:], y_hat[i_right], p_minus, remaining_depth),
        None, # y_pred
    )

def _predict_tree(X, tree):
    """Recursively predict X"""
    if tree.feature is None:
        return tree.y_pred * np.ones(len(X))
    y_pred = np.empty(len(X), dtype=int)
    i_left = X[:,tree.feature] < tree.threshold
    i_right = np.logical_not(i_left)
    y_pred[i_left] = _predict_tree(X[i_left,:], tree.left)
    y_pred[i_right] = _predict_tree(X[i_right,:], tree.right)
    return y_pred
