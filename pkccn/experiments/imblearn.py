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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, RepeatedStratifiedKFold
from tqdm.auto import tqdm

def datasets(is_test_run=False):
    """Yield the names of all evaluated imblearn data sets"""
    if is_test_run:
        return ["yeast_me2", "ecoli"] # small data sets for testing
    return [ # imblearn data sets, sorted by their clean AUROC
        "letter_img",
        "pen_digits",
        "car_eval_4",
        "optical_digits",
        "thyroid_sick",
        "isolet",
        "sick_euthyroid",
        "car_eval_34",
        "spectrometer",
        "protein_homo",
        "libras_move",
        "arrhythmia",
        "webpage",
        "satimage",
        "yeast_me2",
        "mammography",
        "ecoli",
        "us_crime",
        "oil",
        "ozone_level",
        # the following data sets have a clean AUROC below .85:
        # "abalone",
        # "wine_quality",
        # "scene",
        # "solar_flare_m0",
        # "abalone_19",
        # "coil_2000",
        # "yeast_ml8"
    ]

def trial(args, p_minus, p_plus, methods, clf, dataset, X, y):
    """A single trial of imblearn.main()"""
    i_trial, (i_trn, i_tst) = args # unpack the tuple
    y_trn = inject_ccn(y[i_trn], p_minus, p_plus)
    y_pred_trn = cross_val_predict(
        clf,
        X[i_trn,:],
        y_trn,
        method = "predict_proba",
        cv = 5
    )[:,1] # out-of-bag soft predictions
    clf.fit(X[i_trn,:], y_trn) # complete fit
    y_pred_tst = clf.predict_proba(X[i_tst,:])[:,1]

    # use the current model with all thresholding methods
    trial_results = []
    for method_name, method in methods.items():
        threshold = method(y_trn, y_pred_trn)
        y_pred = (y_pred_tst > threshold).astype(int) * 2 - 1 # in [-1, 1]
        trial_results.append({
            "dataset": dataset,
            "method": method_name,
            "trial": i_trial,
            "threshold": threshold,
            "accuracy": accuracy_score(y[i_tst], y_pred),
            "f1": f1_score(y[i_tst], y_pred),
            "lima": lima_score(y[i_tst], y_pred, p_minus),
        })
    return trial_results

def main(
        output_path,
        p_minus,
        p_plus,
        seed = 867,
        n_folds = 10,
        n_repetitions = 10,
        is_test_run = False,
    ):
    print(f"Starting an imblearn experiment to produce {output_path} with seed {seed}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # configure the thresholding methods
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
        "default (accuracy)":
            Threshold("default", metric="accuracy"),
        "default (F1 score)":
            Threshold("default", metric="f1"),
    }

    # set up the base classifier and the repeated cross validation splitter
    clf = RandomForestClassifier()
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repetitions)
    print(f"Each of the {rskf.get_n_splits()} trials evaluates {len(methods)} methods")

    # reduce the experimental grid for testing?
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
        clf = RandomForestClassifier(n_estimators=3)

    # iterate over all data sets
    results = []
    for i_dataset, dataset in enumerate(datasets(is_test_run)):
        imblearn_dataset = fetch_datasets()[dataset]
        X = imblearn_dataset.data
        y = imblearn_dataset.target

        # parallelize over repeated stratified splitting
        with Pool() as pool:
            trial_Xy = partial(trial, p_minus=p_minus, p_plus=p_plus, methods=methods, clf=clf, dataset=dataset, X=X, y=y)
            trial_results = tqdm(
                pool.imap(trial_Xy, enumerate(rskf.split(X, y))),
                desc = f"{dataset} [{i_dataset+1}/{len(datasets(is_test_run))}]",
                total = rskf.get_n_splits(),
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
        lima = ("lima", "mean"),
        lima_std = ("lima", "std"),
    )
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
    parser.add_argument('--n_repetitions', type=int, default=10, metavar='N',
                        help='number of repetitions of the cross validation (default: 10)')
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
