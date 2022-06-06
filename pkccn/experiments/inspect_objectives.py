import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from functools import partial
from imblearn.datasets import fetch_datasets
from multiprocessing import Pool
from pkccn import f1_score, lima_score, Threshold
from pkccn.experiments import MLPClassifier
from pkccn.experiments.imblearn import datasets
from pkccn.data import inject_ccn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

def trial(dataset, fold, trial_seed, n_folds, p_minus, p_plus, clf):
    """A single trial of inspect_objectives.main()"""
    imblearn_dataset = fetch_datasets()[dataset]
    X = imblearn_dataset.data
    y = imblearn_dataset.target
    np.random.seed(trial_seed) # reproduce the imblearn experiment
    y_ccn = inject_ccn(y, p_minus, p_plus)
    i_trn, i_tst = _get_index_of_generator(
        StratifiedKFold(n_folds, shuffle=True).split(X, y),
        fold
    ) # select a single fold
    clf.fit(X[i_trn,:], y_ccn[i_trn])
    y_trn = clf.oob_decision_function_[:,1]
    y_tst = clf.predict_proba(X[i_tst,:])[:,1]

    # evaluate a dense grid of possible thresholds
    results = []
    for t in np.arange(0., 1.001, step=0.001):
        y_t_trn = (y_trn > t).astype(int) * 2 - 1 # in [-1, 1]
        y_t_tst = (y_tst > t).astype(int) * 2 - 1
        results.append({
            "threshold": t,
            #
            # noisy training performances (estimating the clean F1 score)
            "f1_trn_ckccn": _f1_score(y_ccn[i_trn], y_t_trn, y_trn, p_minus, p_plus),
            "f1_trn_pkccn": _f1_score(y_ccn[i_trn], y_t_trn, y_trn, p_minus),
            "f1_trn_cuccn": _f1_score(y_ccn[i_trn], y_t_trn, y_trn),
            "lima_trn": lima_score(y_ccn[i_trn], y_t_trn, p_minus),
            #
            # noisy test performances
            "f1_tst_ckccn": _f1_score(y_ccn[i_tst], y_t_tst, y_tst, p_minus, p_plus),
            "f1_tst_pkccn": _f1_score(y_ccn[i_tst], y_t_tst, y_tst, p_minus),
            "f1_tst_cuccn": _f1_score(y_ccn[i_tst], y_t_tst, y_tst),
            "lima_tst": lima_score(y_ccn[i_tst], y_t_tst, p_minus),
            #
            # clean training and test performances (lima_score is undefined if p_minus==0)
            "f1_trn": f1_score(y[i_trn], y_t_trn),
            "f1_tst": f1_score(y[i_tst], y_t_tst),
        })
    df = pd.DataFrame(results)
    df['p_minus'] = p_minus
    df['p_plus'] = p_plus
    df['dataset'] = dataset
    df['fold'] = fold
    df['trial_seed'] = trial_seed
    return df

def _f1_score(y_hat, y_t, y_pred, p_minus=None, p_plus=None):
    try:
        return f1_score(y_hat, y_t, y_pred, p_minus=p_minus, p_plus=p_plus)
    except ValueError:
        return 0.0

def main(
        output_path,
        p_minus,
        p_plus,
        fold = 0,
        repetition = 0,
        seed = 867,
        n_folds = 10,
        n_repetitions = 20,
        is_test_run = False,
    ):
    print(f"Starting to inspect_objectives, producing {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # configure the base classifier
    clf = RandomForestClassifier(oob_score=True, max_depth=8)

    # parallelize over the data sets
    trial_seed = np.random.randint(np.iinfo(np.uint32).max, size=n_repetitions)[repetition]
    with Pool() as pool:
        trial_dataset = partial(trial, fold=fold, trial_seed=trial_seed, n_folds=n_folds, p_minus=p_minus, p_plus=p_plus, clf=clf)
        results = tqdm(
            pool.imap(trial_dataset, datasets(is_test_run)),
            total = len(datasets(is_test_run)),
            ncols = 80
        )
        df = pd.concat(results)

    # store one file per data set
    for dataset, dataset_df in df.groupby("dataset", sort=False):
        dataset_path = output_path.replace(".csv", f"_{dataset}.csv")
        dataset_df.to_csv(dataset_path)
        print(f"{dataset_df.shape[0]} results succesfully stored at {dataset_path}")
    df.to_csv(output_path) # also store a file with all data sets
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

def _get_index_of_generator(generator, index):
    """Get generator[index] when the generator does not allow indexing."""
    for i, item in enumerate(generator):
        if i != index:
            return item

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('p_minus', type=float, help='noise rate of the negative class')
    parser.add_argument('p_plus', type=float, help='noise rate of the positive class')
    parser.add_argument('--fold', type=int, default=0, metavar='N',
                        help='the fold to plot (default: 0)')
    parser.add_argument('--repetition', type=int, default=0, metavar='N',
                        help='the repetition to plot (default: 0)')
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
        args.fold,
        args.repetition,
        args.seed,
        args.n_folds,
        args.n_repetitions,
        args.is_test_run,
    )
