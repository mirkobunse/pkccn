import argparse
import numpy as np
import os
import pandas as pd
from functools import partial
from multiprocessing import Pool
from pkccn import lima_score, Threshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

def read_fact(dl2_path="data/fact_dl2.hdf5", dl3_path="data/fact_dl3.hdf5"):
    """TODO"""
    X = None
    y_hat = None
    return X, y_hat

def trial(trial_seed, n_folds, methods, clf, X, y_hat):
    """A single trial of imblearn.main()"""
    np.random.seed(trial_seed)

    # cross_val_predict, fitting a separate threshold in each fold
    y_pred = { m: np.zeros_like(y_hat) for m in methods.keys() } # method name -> predictions
    thresholds = { m: [] for m in methods.keys() } # method name -> thresholds
    for i_trn, i_tst in StratifiedKFold(n_folds, shuffle=True).split(X, y_hat):
        clf.fit(X[i_trn,:], y_hat[i_trn])
        y_trn = clf.oob_decision_function_[:,1]
        y_tst = clf.predict_proba(X[i_tst,:])[:,1]
        for method_name, method in methods.items():
            threshold = method(y_hat[i_trn], y_trn)
            y_pred[method_name][i_tst] = (y_tst > threshold).astype(int) * 2 - 1 # in [-1, 1]
            thresholds[method_name].append(threshold)

    # evaluate all predictions
    trial_results = []
    for method_name, y_method in y_pred.items():
        trial_results.append({
            "method": method_name,
            "threshold": np.mean(thresholds[method_name]),
            "trial_seed": trial_seed,
            "lima": -1, # TODO
        })
    return trial_results

def main(
        output_path,
        seed = 867,
        n_folds = 10,
        n_repetitions = 5,
        fake_labels = False,
        is_test_run = False,
    ):
    print(f"Starting a fact experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)
    p_minus = 1 / 6 # this value is equivalent to alpha = 1 / 5

    # configure the thresholding methods, the base classifier, and read the noisy data
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
    X, y_hat = read_fact() # TODO

    # parallelize over repetitions
    results = []
    trial_seeds = np.random.randint(np.iinfo(np.uint32).max, size=n_repetitions)
    with Pool() as pool:
        trial_Xy = partial(trial, n_folds=n_folds, methods=methods, clf=clf, X=X, y_hat=y_hat)
        trial_results = tqdm(
            pool.imap(trial_Xy, trial_seeds), # each trial gets a different seed
            total = n_repetitions,
            ncols = 80
        )
        for result in trial_results:
            results.extend(result)

    # aggregate and store the results
    df = pd.DataFrame(results)
    df = df.groupby("method", sort=False).agg(
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
    parser.add_argument('--n_folds', type=int, default=10, metavar='N',
                        help='number of cross validation folds (default: 10)')
    parser.add_argument('--n_repetitions', type=int, default=5, metavar='N',
                        help='number of repetitions of the cross validation (default: 5)')
    parser.add_argument("--fake_labels", action="store_true")
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.seed,
        args.n_folds,
        args.n_repetitions,
        args.fake_labels,
        args.is_test_run,
    )
