import argparse
import numpy as np
import os
import pandas as pd
from imblearn.datasets import fetch_datasets
from pkccn import Threshold
from pkccn.data import inject_ccn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit

def main(
        output_path,
        dataset,
        p_minus,
        p_plus,
        seed = 867,
        n_trials = 10,
        n_folds = 5,
    ):
    print(f"Starting an imblearn experiment to produce {output_path} with seed {seed}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # read the data
    imblearn_dataset = fetch_datasets()[dataset]
    X = imblearn_dataset.data
    y = imblearn_dataset.target

    # set up the base classifier
    clf = RandomForestClassifier(n_jobs=-1)

    # configure the thresholding methods
    methods = {
        "Menon et al. (2015; CU-CCN; accuracy)": Threshold("menon"),
        "Mithal et al. (2017; CU-CCN; G-measure)": Threshold("mithal"),
        "Menon et al. (2015; PK-CCN; accuracy)": Threshold("menon", p_minus=p_minus),
        "Menon et al. (2015; CK-CCN; accuracy)": Threshold("menon", p_minus=p_minus, p_plus=p_plus),
        "default (accuracy)": Threshold("default", metric="accuracy"),
    }

    # repeated stratified splitting
    results = []
    for trial, (i_trn, i_tst) in enumerate(StratifiedShuffleSplit(n_trials, test_size=.5).split(X, y)):
        y_trn = inject_ccn(y[i_trn], p_minus, p_plus, random_state=trial)
        y_pred_trn = cross_val_predict(clf, X[i_trn,:], y_trn, method="predict_proba", cv=n_folds)[:,1]
        clf.fit(X[i_trn,:], y_trn) # complete fit
        y_pred_tst = clf.predict_proba(X[i_tst,:])[:,1]

        # use the current model with all thresholding methods
        for method_name, method in methods.items():
            threshold = method(y_trn, y_pred_trn)
            y_pred = (y_pred_tst > threshold).astype(int) * 2 - 1 # in [-1, 1]
            results.append({
                "method": method_name,
                "trial": trial,
                "threshold": threshold,
                "accuracy": accuracy_score(y[i_tst], y_pred),
                "f1": f1_score(y[i_tst], y_pred),
            })

    # aggregate and store the results
    df = pd.DataFrame(results)
    df = df.groupby("method", sort=False).agg(
        threshold = ("threshold", "mean"),
        threshold_std = ("threshold", "std"),
        accuracy = ("accuracy", "mean"),
        accuracy_std = ("accuracy", "std"),
        f1 = ("f1", "mean"),
        f1_std = ("f1", "std"),
    )
    df['dataset'] = dataset
    df['p_minus'] = p_minus
    df['p_plus'] = p_plus
    df.to_csv(output_path)
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('dataset', type=str, help='name of an imblearn data set')
    parser.add_argument('p_minus', type=float, help='noise rate of the negative class')
    parser.add_argument('p_plus', type=float, help='noise rate of the positive class')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument('--n_trials', type=int, default=10, metavar='N',
                        help='number of trials (default: 10)')
    parser.add_argument('--n_folds', type=int, default=5, metavar='N',
                        help='number of cross validation folds (default: 5)')
    args = parser.parse_args()
    main(
        args.output_path,
        args.dataset,
        args.p_minus,
        args.p_plus,
        args.seed,
        args.n_trials,
        args.n_folds,
    )
