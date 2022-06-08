import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from functools import partial
from imblearn.datasets import fetch_datasets
from multiprocessing import Pool
from pkccn import lima_score, Threshold
from pkccn.data import inject_ccn
from pkccn.experiments.imblearn import datasets
from pkccn.tree import LiMaRandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

def _store(results, path, p_minus, p_plus):
    """Aggregate and store the results"""
    df = pd.DataFrame(results)
    df = df.groupby(["dataset", "method"], sort=False).agg(
        threshold = ("threshold", "mean"),
        threshold_std = ("threshold", "std"),
        accuracy = ("accuracy", "mean"),
        accuracy_std = ("accuracy", "std"),
        f1 = ("f1", "mean"),
        f1_std = ("f1", "std"),
        lima = ("lima", "mean"),
        lima_std = ("lima", "std"),
    )
    print(df.groupby("method", sort=False).agg(f1 = ("f1", "mean")))
    df['p_minus'] = p_minus
    df['p_plus'] = p_plus
    df.to_csv(path)
    print(f"{df.shape[0]} results succesfully stored at {path}")

def main(
        output_path,
        p_minus,
        p_plus,
        seed = 867,
        n_folds = 10,
        n_repetitions = 20,
        is_test_run = False,
    ):
    print(f"Starting an imblearn_tree experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # configure the Li&Ma classifier
    method_name = "Li \& Ma tree (ours; PK-CCN)"
    clf = LiMaRandomForest(p_minus, max_depth=8, n_jobs=-1)
    if is_test_run:
        clf.max_depth = 4
        clf.n_estimators = 32

    # iterate over all repetitions; parallelize via LiMaRandomForest
    results = []
    trial_seeds = np.random.randint(np.iinfo(np.uint32).max, size=n_repetitions)
    for i_trial, trial_seed in enumerate(trial_seeds):
        progressbar = tqdm(
            datasets(is_test_run), # iterate over all data sets
            desc = f"Repetition {i_trial+1}/{len(trial_seeds)}",
            total = len(datasets(is_test_run)),
            ncols = 80
        )
        for dataset in progressbar:
            imblearn_dataset = fetch_datasets()[dataset]
            X = imblearn_dataset.data
            y = imblearn_dataset.target
            np.random.seed(trial_seed)
            y_ccn = inject_ccn(y, p_minus, p_plus)

            # cross_val_predict, fitting a separate model in each fold
            y_pred = np.zeros_like(y)
            thresholds = []
            for i_trn, i_tst in StratifiedKFold(n_folds, shuffle=True).split(X, y):
                clf.fit(X[i_trn,:], y_ccn[i_trn])
                y_pred[i_tst] = clf.predict(X[i_tst,:])
                thresholds.append(clf.threshold)
            results.append({
                "dataset": dataset,
                "method": method_name,
                "threshold": np.mean(thresholds),
                "trial_seed": trial_seed,
                "accuracy": accuracy_score(y, y_pred),
                "f1": f1_score(y, y_pred),
                "lima": lima_score(y_ccn, y_pred, p_minus), # noisy LiMa
            })
        _store(results, output_path.replace(".csv", ".tmp.csv"), p_minus, p_plus)
    _store(results, output_path, p_minus, p_plus) # store all results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('p_minus', type=float, help='noise rate of the negative class')
    parser.add_argument('p_plus', type=float, help='noise rate of the positive class')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument('--n_folds', type=int, default=10, metavar='N',
                        help='number of cross validation folds (default: 10)')
    parser.add_argument('--n_repetitions', type=int, default=20, metavar='N',
                        help='number of repetitions of the cross validation (default: 20)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.p_minus,
        args.p_plus,
        args.seed,
        args.n_folds,
        args.n_repetitions,
        args.is_test_run,
    )
