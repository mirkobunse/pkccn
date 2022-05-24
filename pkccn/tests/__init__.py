import numpy as np
from imblearn.datasets import fetch_datasets
from pkccn import ThresholdedClassifier
from pkccn.data import inject_ccn
from sklearn.linear_model import LogisticRegression
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
        print(f"method=\"{method}\" achieves accuracy={accuracy}")
    def test_lima(self):
        self.__test_method(
            "lima",
            method_args = {"random_state": RANDOM_STATE, "p_minus": .5}
        )
    def test_default(self):
        self.__test_method("default")
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
