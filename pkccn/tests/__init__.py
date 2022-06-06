import numpy as np
from imblearn.datasets import fetch_datasets
from fact.analysis.statistics import li_ma_significance
from pkccn import f1_score, lima_score, Threshold, ThresholdedClassifier
from pkccn.experiments import MLPClassifier
from pkccn.data import inject_ccn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import balanced_accuracy_score
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

class TestMLPClassifier(TestCase):
    def test_classifier(self):
        X_trn, X_tst, y_trn, y_tst = fetch_data("coil_2000")
        clf = MLPClassifier(class_weight="balanced")
        clf.fit(X_trn, y_trn)
        print(f"balanced accuracy: {balanced_accuracy_score(y_tst, clf.predict(X_tst))}")

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
    def test_objectives(self):
        rng = np.random.RandomState(RANDOM_STATE)
        y_true = rng.choice([-1, 1], size=100000, p=[.8, .2])
        y_pred = rng.rand(100000) * y_true + rng.randn(100000) * .5 # synthetic predictions
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

        y_hat = inject_ccn(y_true, .5, .1, random_state=RANDOM_STATE) # CCN noise
        p_plus = np.sum((y_hat == -1) & (y_true == 1)) / np.sum(y_true == 1)
        p_minus = np.sum((y_hat == 1) & (y_true == -1)) / np.sum(y_true == -1)
        # print(f"p_+={p_plus}, p_-={p_minus}")

        for threshold in rng.rand(20):
            y_threshold = (y_pred > threshold).astype(int) * 2 - 1
            theirs = sklearn_f1_score(y_true, y_threshold)
            ours = f1_score(y_true, y_threshold)
            self.assertAlmostEqual(theirs, ours) # clean labels

            # F1 under label noise
            menon = f1_score(y_hat, y_threshold, p_minus=p_minus, p_plus=p_plus)
            self.assertAlmostEqual(ours, menon, delta=.05) # noisy labels

            # lima_score (ours) vs li_ma_significance (gamma ray astronomy)
            alpha = p_minus / (1 - p_minus)
            _y_hat = y_hat[y_pred > threshold]
            N = len(_y_hat)
            N_plus = np.sum(_y_hat == 1)
            N_minus = N - N_plus
            theirs = li_ma_significance(N_plus, N_minus, alpha)
            ours = lima_score(y_hat, y_threshold, p_minus)
            self.assertAlmostEqual(theirs, ours)

class TestOptimizations(TestCase):
    def test_optimizations(self):
        rng = np.random.RandomState(RANDOM_STATE)
        for p in rng.rand(10):
            y_true = rng.choice([-1, 1], size=10000, p=[p, 1-p])
            y_pred = rng.rand(10000) * y_true + rng.randn(10000) * .5 # synthetic predictions
            y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

            # find the optimal threshold via grid search (inefficient)
            T = np.arange(0.001, 1., step=.001) # threshold grid
            f1_T = np.array([sklearn_f1_score(y_true, (y_pred > t).astype(int) * 2 - 1) for t in T])
            best_i = np.argmin(f1_T) # index of best threshold / F1 score

            # compare with a numerically optimized threshold (efficient)
            t = Threshold("default", metric="f1", random_state=RANDOM_STATE)(y_true, y_pred)
            f1_t = sklearn_f1_score(y_true, (y_pred > t).astype(int) * 2 - 1)
            if f1_T[best_i] > f1_t:
                self.assertAlmostEqual(f1_T[best_i], f1_t) # compare scores

            # CCN noise with CK-CCN menon_threshold
            y_hat = inject_ccn(y_true, .5, .1, random_state=RANDOM_STATE)
            p_plus = np.sum((y_hat == -1) & (y_true == 1)) / np.sum(y_true == 1)
            p_minus = np.sum((y_hat == 1) & (y_true == -1)) / np.sum(y_true == -1)
            m = Threshold("menon", metric="f1", p_minus=p_minus, p_plus=p_plus, random_state=RANDOM_STATE)(y_hat, y_pred)
            f1_m = sklearn_f1_score(y_true, (y_pred > m).astype(int) * 2 - 1)
            self.assertAlmostEqual(f1_t, f1_m, delta=1e-2) # compare scores

            # lima_score
            lima_T = np.array([lima_score(y_hat, (y_pred > t).astype(int) * 2 - 1, p_minus=p_minus) for t in T])
            best_i = np.argmin(lima_T) # grid search (inefficient)
            t = Threshold("lima", p_minus=p_minus, random_state=RANDOM_STATE)(y_hat, y_pred) # numeric (efficient)
            lima_t = lima_score(y_hat, (y_pred > t).astype(int) * 2 - 1, p_minus=p_minus)
            if lima_T[best_i] > lima_t:
                self.assertAlmostEqual(lima_T[best_i], lima_t) # compare scores

class TestThresholdedClassifier(TestCase):
    def _test_method(self, method, p_minus=.5, p_plus=.1, method_args={}):
        print() # empty line to go beyond a leading "."
        X_trn, X_tst, y_trn, y_tst = fetch_data()
        y_trn = inject_ccn(y_trn, p_minus, p_plus, random_state=RANDOM_STATE)
        clf = ThresholdedClassifier(
            RandomForestClassifier(max_depth=4, class_weight="balanced", oob_score=True, random_state=RANDOM_STATE),
            method,
            prediction_method = "oob",
            method_args = {"verbose": True, **method_args}
        )
        clf.fit(X_trn, y_trn)
        accuracy = clf.score(X_tst, y_tst)
        f1 = sklearn_f1_score(y_tst, clf.predict(X_tst))
        print(f"method=\"{method}\" achieves CCN accuracy={accuracy:.3f}, f1={f1:.3f}")
    def test_lima(self):
        self._test_method(
            "lima",
            method_args = {"random_state": RANDOM_STATE, "p_minus": .5}
        )
    def test_default_accuracy(self):
        self._test_method("default")
    def test_default_f1(self):
        self._test_method(
            "default",
            method_args = {"metric": "f1"}
        )
    def test_menon_accuracy(self):
        self._test_method("menon")
    def test_menon_f1(self):
        self._test_method(
            "menon",
            method_args = {"metric": "f1"}
        )
    def test_ckccn_menon_f1(self):
        self._test_method(
            "menon",
            method_args = {"metric": "f1", "p_minus": .5, "p_plus": .1}
        )
    def test_pkccn_menon(self):
        self._test_method(
            "menon",
            method_args = {"p_minus": .5}
        )
    def test_ckccn_menon(self):
        self._test_method(
            "menon",
            method_args = {"p_minus": .5, "p_plus": .1}
        )
    def test_mithal(self):
        self._test_method(
            "mithal",
            method_args = {"random_state": RANDOM_STATE}
        )
    def test_yao(self):
        self._test_method("yao")
