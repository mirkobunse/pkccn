import numpy as np
from imblearn.datasets import fetch_datasets
from pkccn import __f1_objective as _TestObjectives__f1_objective # unittest name mangling
from pkccn import ThresholdedClassifier
from pkccn.data import inject_ccn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from unittest import TestCase

RANDOM_STATE = 876

def fetch_data(name='ecoli', random_state=RANDOM_STATE):
    dataset = fetch_datasets()[name]
    X_trn, X_tst, y_trn, y_tst = train_test_split(
        dataset.data,
        dataset.target,
        test_size = 0.5,
        random_state = random_state,
        stratify = dataset.target
    )
    return (X_trn, X_tst, y_trn, y_tst)

class TestData(TestCase):
    def test_inject_ccn(self):
        X_trn, X_tst, y_trn, y_tst = fetch_data()
        y_hat1 = inject_ccn(y_trn, .5, .1, random_state=RANDOM_STATE)
        self.assertTrue((y_trn != y_hat1).any()) # not equal
        y_hat2 = inject_ccn(y_trn, .5, .1, random_state=RANDOM_STATE) # same RandomState
        self.assertTrue((y_hat1 == y_hat2).all()) # equal
        y_hat2 = inject_ccn(y_trn, .5, .1, random_state=RANDOM_STATE+1) # different RandomState
        self.assertTrue((y_hat1 != y_hat2).any()) # not equal

class TestObjectives(TestCase):
    def test_f1_objective(self):
        rng = np.random.RandomState(RANDOM_STATE)
        y_true = rng.choice([-1, 1], size=1000, p=[.8, .2])
        p = np.sum(y_true == 1) / len(y_true) # class 1 prevalence
        y_pred = rng.rand(1000) * y_true + rng.randn(1000) * .5 # synthetic predictions
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
        for threshold in rng.rand(100):
            y_threshold = (y_pred > threshold).astype(int) * 2 - 1
            theirs = f1_score(y_true, y_threshold)
            ours = - __f1_objective(threshold, y_true, y_pred, p)
            self.assertAlmostEqual(theirs, ours)

class TestThresholdedClassifier(TestCase):
    def __test_method(self, method, p_minus=.5, p_plus=.1, method_args={}):
        print() # empty line to go beyond a leading "."
        X_trn, X_tst, y_trn, y_tst = fetch_data()
        y_trn = inject_ccn(y_trn, p_minus, p_plus, random_state=RANDOM_STATE)
        clf = ThresholdedClassifier(
            LogisticRegression(random_state=RANDOM_STATE),
            method,
            method_args = {"verbose": True, **method_args}
        )
        clf.fit(X_trn, y_trn)
        accuracy = clf.score(X_tst, y_tst)
        f1 = f1_score(y_tst, clf.predict(X_tst))
        print(f"method=\"{method}\" achieves CCN accuracy={accuracy:.3f}, f1={f1:.3f}")
    def test_lima(self):
        self.__test_method(
            "lima",
            method_args = {"random_state": RANDOM_STATE, "p_minus": .5}
        )
    def test_default_accuracy(self):
        self.__test_method("default")
    def test_default_f1(self):
        self.__test_method(
            "default",
            method_args = {"metric": "f1"}
        )
    def test_menon(self):
        self.__test_method("menon")
    def test_pkccn_menon(self):
        self.__test_method(
            "menon",
            method_args = {"p_minus": .5}
        )
    def test_ckccn_menon(self):
        self.__test_method(
            "menon",
            method_args = {"p_minus": .5, "p_plus": .1}
        )
    def test_mithal(self):
        self.__test_method(
            "mithal",
            method_args = {"random_state": RANDOM_STATE}
        )
