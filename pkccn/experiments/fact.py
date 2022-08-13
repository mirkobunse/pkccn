import argparse
import numpy as np
import os
import pandas as pd
from functools import partial
from multiprocessing import Pool
from pkccn import lima_score, Threshold
from pkccn.tree import LiMaRandomForest
from fact.io import read_data as fact_read_data
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

FEATURES = [ # what to read from the FACT HDF5 files
    "concentration_cog",
    "concentration_core",
    "concentration_one_pixel",
    "concentration_two_pixel",
    "leakage1",
    "leakage2",
    "size",
    "width",
    "length",
    "skewness_long",
    "skewness_trans",
    "kurtosis_long",
    "kurtosis_trans",
    "num_islands",
    "num_pixel_in_shower",
    "photoncharge_shower_variance",
    "area",
    "log_size",
    "size_area",
    "area_size_cut_var"
]

def _extract_weak_labels(X, df, theta2_cut=0.025):
    """Extract noisy On/Off region labels from a DataFrame."""
    theta_cut = np.sqrt(theta2_cut)
    is_on = (df.theta_deg < theta_cut).values
    is_off = pd.concat([df[f'theta_deg_off_{i}'] < theta_cut for i in range(1, 6)], axis=1).values
    is_labeled = np.logical_or(is_on, np.any(is_off, axis=1)) # only consider labeled instances
    day = pd.to_datetime(df['timestamp_y'], unit="s").dt.dayofyear
    group = LabelEncoder().fit_transform(day.values.reshape(-1, 1))
    X = X[is_labeled]
    y_hat = is_on[is_labeled] * 2 - 1
    group = group[is_labeled]
    return X, y_hat, group

def _replace_on_position(dl3, off_position=1):
    """Replace the On region with one of the Off regions."""
    column = f"theta_deg_off_{off_position}"
    dl3["theta_deg"] = dl3[column]
    dl3[column] = np.inf # no event is considered in this region
    return dl3

def _read_fact(fake_labels, dl2_path, dl3_path):
    """Load real-world data. Joins DL2 and DL3 files and computes auxiliary features."""
    dl3 = fact_read_data(dl3_path, "events")
    dl2 = fact_read_data(dl2_path, "events")
    dl3 = dl3.rename({"event_num": "event"}, axis='columns')
    dl2 = dl2.rename({"event_num": "event"}, axis='columns')
    dl3 = dl3.merge(dl2, on=["event", "night", "run_id"], how="inner")
    if fake_labels:
        dl3 = _replace_on_position(dl3)
    return dl3

def read_noisy(fake_labels=False, dl2_path="data/fact_dl2.hdf5", dl3_path="data/fact_dl3.hdf5"):
    """Load real-world data with noisy labels."""
    dl3 = _read_fact(fake_labels, dl2_path, dl3_path)
    dl3["area"] = dl3["width"] * dl3["length"] * np.pi
    dl3["log_size"] = np.log(dl3["size"])
    dl3["size_area"] = dl3["size"] / dl3["area"]
    dl3["area_size_cut_var"] = dl3["area"] / (dl3["log_size"] ** 2)
    return _extract_weak_labels(dl3[FEATURES].values, dl3)

def read_sota(fake_labels=False, dl2_path="data/fact_dl2.hdf5", dl3_path="data/fact_dl3.hdf5"):
    """Load real-world data with SOTA and noisy labels."""
    dl3 = _read_fact(fake_labels, dl2_path, dl3_path)
    return _extract_weak_labels(dl3[["gamma_prediction"]].values, dl3)

def trial(trial_seed, methods, clf, X, y_hat, group, p_minus):
    """A single trial of fact.main()"""
    np.random.seed(trial_seed)

    # cross_val_predict, fitting a separate threshold in each fold
    y_pred = { m: np.zeros_like(y_hat) for m in methods.keys() } # method name -> predictions
    thresholds = { m: [] for m in methods.keys() } # method name -> thresholds
    for i_trn, i_tst in GroupKFold(len(np.unique(group))).split(X, y_hat, group):
        clf.fit(X[i_trn,:], y_hat[i_trn])
        if hasattr(clf, "oob_decision_function_"):
            y_trn = clf.oob_decision_function_[:,1]
        else:
            y_trn = clf.predict_proba(X[i_trn, :])[:,1]
        y_tst = clf.predict_proba(X[i_tst, :])[:,1]
        for method_name, method in methods.items():
            threshold = method(y_hat[i_trn], y_trn)
            y_pred[method_name][i_tst] = (y_tst > threshold).astype(int) * 2 - 1 # in [-1, 1]
            thresholds[method_name].append(threshold)

    # evaluate all predictions
    trial_results = []
    for method_name, y_method in y_pred.items():
        trial_results.append({
            "classifier": type(clf).__name__,
            "method": method_name,
            "threshold": np.mean(thresholds[method_name]),
            "trial_seed": trial_seed,
            "lima": lima_score(y_hat, y_method, p_minus),
        })
    return trial_results

class SotaClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        return np.stack((1-X[:, 0], X[:, 0])).T
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def main(
        output_path,
        seed = 867,
        n_repetitions = 20,
        fake_labels = False,
        is_test_run = False,
    ):
    print(f"Starting a fact experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)
    p_minus = 1 / 6 # this value is equivalent to alpha = 1 / 5
    if fake_labels:
        p_minus = 1 / 5

    # configure the thresholding methods and the base classifier
    methods = {
        "Li \& Ma threshold (ours; PK-CCN)":
            Threshold("lima", p_minus=p_minus),
        "Menon et al. (2015; PK-CCN; F1 score)":
            Threshold("menon", metric="f1", p_minus=p_minus),
        "Menon et al. (2015; CU-CCN; F1 score)":
            Threshold("menon", metric="f1"),
        "Mithal et al. (2017; CU-CCN; G measure)":
            Threshold("mithal"),
        "default (F1 score)":
            Threshold("default", metric="f1"),
    }

    clf = RandomForestClassifier(oob_score=True, max_depth=8)
    sota_clf = SotaClassifier()

    # read the noisy data
    print("Loading the data...")
    X, y_hat, group = read_noisy(fake_labels)
    X_sota, _, _ = read_sota(fake_labels)
    print(f"Read the data of {len(np.unique(group))} days to cross-validate over")

    # experiment with thresholding methods: parallelize over repetitions
    results = []
    trial_seeds = np.random.randint(np.iinfo(np.uint32).max, size=n_repetitions)
    with Pool() as pool:
        trial_Xyg = partial(trial, methods=methods, clf=sota_clf, X=X_sota, y_hat=y_hat, group=group, p_minus=p_minus)
        trial_results = tqdm(
            pool.imap(trial_Xyg, trial_seeds), # each trial gets a different seed
            desc = f"Thresholding (Clean)",
            total = n_repetitions,
            ncols = 80
        )
        for result in trial_results:
            results.extend(result)
        trial_Xyg = partial(trial, methods=methods, clf=clf, X=X, y_hat=y_hat, group=group, p_minus=p_minus)
        trial_results = tqdm(
            pool.imap(trial_Xyg, trial_seeds), # each trial gets a different seed
            desc = f"Thresholding (Noisy)",
            total = n_repetitions,
            ncols = 80
        )
        for result in trial_results:
            results.extend(result)

    # experiment with the Li&Ma tree: parallelize over ensemble members
    clf = LiMaRandomForest(p_minus, max_depth=8, n_jobs=-1)
    if is_test_run:
        clf.max_depth = 2
        clf.n_estimators = 32
    progressbar = tqdm(
        trial_seeds, # use the same seeds as above
        desc = f"Li & Ma tree",
        total = n_repetitions,
        ncols = 80
    )
    for trial_seed in progressbar:
        np.random.seed(trial_seed)
        y_pred = np.zeros_like(y_hat)
        thresholds = []
        for i_trn, i_tst in GroupKFold(len(np.unique(group))).split(X, y_hat, group):
            clf.fit(X[i_trn,:], y_hat[i_trn])
            y_pred[i_tst] = clf.predict(X[i_tst,:])
            thresholds.append(clf.threshold)
        results.append({
            "classifier": type(clf).__name__,
            "method": "Li \& Ma tree (ours; PK-CCN)",
            "threshold": np.mean(thresholds),
            "trial_seed": trial_seed,
            "lima": lima_score(y_hat, y_pred, p_minus),
        })

    # aggregate and store the results
    df = pd.DataFrame(results)
    df = df.groupby(["method", "classifier"], sort=False).agg(
        threshold = ("threshold", "mean"),
        threshold_std = ("threshold", "std"),
        lima = ("lima", "mean"),
        lima_std = ("lima", "std"),
    )
    print(df)
    df.to_csv(output_path)
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument('--n_repetitions', type=int, default=20, metavar='N',
                        help='number of repetitions of the cross validation (default: 20)')
    parser.add_argument("--fake_labels", action="store_true")
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.seed,
        args.n_repetitions,
        args.fake_labels,
        args.is_test_run,
    )
