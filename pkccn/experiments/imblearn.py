import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from functools import partial
from imblearn.datasets import fetch_datasets
from multiprocessing import Pool
from pkccn import g_score, lima_score, Threshold
from pkccn.data import inject_ccn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

def datasets(is_test_run=False):
    """Yield the names of all evaluated imblearn data sets"""
    if is_test_run:
        return ["yeast_me2", "ecoli"] # small data sets for testing
    return list(fetch_datasets().keys())

def trial(trial_seed, n_folds, p_minus, p_plus, methods, clf, dataset, X, y):
    """A single trial of imblearn.main()"""
    np.random.seed(trial_seed)
    y_ccn = inject_ccn(y, p_minus, p_plus)

    # cross_val_predict, fitting a separate threshold in each fold
    y_pred = { m: np.zeros_like(y) for m in methods.keys() } # method name -> predictions
    thresholds = { m: [] for m in methods.keys() } # method name -> thresholds
    for i_trn, i_tst in StratifiedKFold(n_folds, shuffle=True).split(X, y):
        clf.fit(X[i_trn,:], y_ccn[i_trn])
        y_trn = clf.oob_decision_function_[:,1]
        y_tst = clf.predict_proba(X[i_tst,:])[:,1]
        for method_name, method in methods.items():
            threshold = method(y_ccn[i_trn], y_trn)
            y_pred[method_name][i_tst] = (y_tst > threshold).astype(int) * 2 - 1 # in [-1, 1]
            thresholds[method_name].append(threshold)

    # evaluate all predictions
    trial_results = []
    for method_name, y_method in y_pred.items():
        trial_results.append({
            "dataset": dataset,
            "method": method_name,
            "threshold": np.mean(thresholds[method_name]),
            "trial_seed": trial_seed,
            "accuracy": accuracy_score(y, y_method),
            "f1": f1_score(y, y_method),
            "g": g_score(y, y_method),
            "lima": lima_score(y_ccn, y_method, p_minus), # noisy LiMa
        })
    return trial_results

def main(
        output_path,
        p_minus,
        p_plus,
        seed = 867,
        n_folds = 10,
        n_repetitions = 20,
        is_test_run = False,
    ):
    print(f"Starting an imblearn experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # configure the thresholding methods and the base classifier
    methods = {
        "Li \& Ma threshold (ours; PK-CCN)":
            Threshold("lima", p_minus=p_minus),
        "Menon et al. (2015; PK-CCN; accuracy)":
            Threshold("menon", p_minus=p_minus),
        "Menon et al. (2015; PK-CCN; F1 score)":
            Threshold("menon", metric="f1", p_minus=p_minus),
        "Menon et al. (2015; CK-CCN; accuracy)":
            Threshold("menon", p_minus=p_minus, p_plus=p_plus),
        "Menon et al. (2015; CK-CCN; F1 score)":
            Threshold("menon", metric="f1", p_minus=p_minus, p_plus=p_plus),
        "Menon et al. (2015; CU-CCN; accuracy)":
            Threshold("menon"),
        "Menon et al. (2015; CU-CCN; F1 score)":
            Threshold("menon", metric="f1"),
        "Mithal et al. (2017; CU-CCN; G measure)":
            Threshold("mithal"),
        "Yao et al. (2020; CU-CCN; accuracy)":
            Threshold("yao"),
        "default (accuracy)":
            Threshold("default", metric="accuracy"),
        "default (F1 score)":
            Threshold("default", metric="f1"),
    }
    clf = RandomForestClassifier(oob_score=True, max_depth=6)

    # iterate over all data sets
    results = []
    trial_seeds = np.random.randint(np.iinfo(np.uint32).max, size=n_repetitions)
    for i_dataset, dataset in enumerate(datasets(is_test_run)):
        imblearn_dataset = fetch_datasets()[dataset]
        X = imblearn_dataset.data
        y = imblearn_dataset.target

        # parallelize over repetitions
        with Pool() as pool:
            trial_Xy = partial(trial, n_folds=n_folds, p_minus=p_minus, p_plus=p_plus, methods=methods, clf=clf, dataset=dataset, X=X, y=y)
            trial_results = tqdm(
                pool.imap(trial_Xy, trial_seeds), # each trial gets a different seed
                desc = f"{dataset} [{i_dataset+1}/{len(datasets(is_test_run))}]",
                total = n_repetitions,
                ncols = 80
            )
            for result in trial_results:
                results.extend(result)

    # aggregate and store the results
    df = pd.DataFrame(results)
    df = df.groupby(["dataset", "method"], sort=False).agg(
        threshold = ("threshold", "mean"),
        threshold_std = ("threshold", "std"),
        accuracy = ("accuracy", "mean"),
        accuracy_std = ("accuracy", "std"),
        f1 = ("f1", "mean"),
        f1_std = ("f1", "std"),
        g = ("g", "mean"),
        g_std = ("g", "std"),
        lima = ("lima", "mean"),
        lima_std = ("lima", "std"),
    )
    print(df.groupby("method", sort=False).agg(f1=("f1", "mean"), g=("g", "mean"), lima=("lima", "mean")))
    df['p_minus'] = p_minus
    df['p_plus'] = p_plus
    df.to_csv(output_path)
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

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
