import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from functools import partial
from imblearn.datasets import fetch_datasets
from multiprocessing import Pool
from pkccn import f1_score, lima_score, Threshold
from pkccn.data import inject_ccn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, RepeatedStratifiedKFold
from tqdm.auto import tqdm

def main(
        output_path,
        p_minus,
        p_plus,
        dataset,
        trial = 0,
        seed = 867,
        n_folds = 10,
        n_repetitions = 10,
    ):
    print(f"Starting to inspect_objectives, producing {output_path} with seed {seed}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # set up the base classifier, the repeated cross validation splitter, and the data
    clf = RandomForestClassifier()
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repetitions)
    imblearn_dataset = fetch_datasets()[dataset]
    X = imblearn_dataset.data
    y = imblearn_dataset.target

    # repeated stratified splitting
    for i_trial, (i_trn, i_tst) in enumerate(rskf.split(X, y)):
        if i_trial != trial:
            continue # advance to the desired trial index
        y_trn = inject_ccn(y[i_trn], p_minus, p_plus)
        y_tst = inject_ccn(y[i_tst], p_minus, p_plus)
        y_pred_trn = cross_val_predict(
            clf,
            X[i_trn,:],
            y_trn,
            method = "predict_proba",
            cv = 5
        )[:,1] # out-of-bag soft predictions
        clf.fit(X[i_trn,:], y_trn) # complete fit
        y_pred_tst = clf.predict_proba(X[i_tst,:])[:,1]

        # inspect eta_min and eta_max
        eta_min_trn, eta_max_trn = np.quantile(y_pred_trn, [.01, .99])
        eta_min_tst, eta_max_tst = np.quantile(y_pred_tst, [.01, .99])
        print(
            f"eta: training {eta_min_trn, eta_max_trn},",
            f"test {eta_min_tst, eta_max_tst}",
        )

        # evaluate a dense grid of thresholds
        results = []
        for t in np.arange(0., 1.001, step=0.001):
            y_t_trn = (y_pred_trn > t).astype(int) * 2 - 1 # in [-1, 1]
            y_t_tst = (y_pred_tst > t).astype(int) * 2 - 1
            results.append({
                "threshold": t,
                "accuracy_trn_hat": accuracy_score(y_trn, y_t_trn), # noisy training performances
                "f1_trn_hat": f1_score(y_trn, y_t_trn, p_minus=p_minus, p_plus=p_plus), # estimate F1 under noise
                "lima_trn_hat": lima_score(y_trn, y_t_trn, p_minus),
                "accuracy_tst_hat": accuracy_score(y_tst, y_t_tst), # noisy test performances
                "f1_tst_hat": f1_score(y_tst, y_t_tst, p_minus=p_minus, p_plus=p_plus),
                "lima_tst_hat": lima_score(y_tst, y_t_tst, p_minus),
                "accuracy_trn": accuracy_score(y[i_trn], y_t_trn), # clean training performances
                "f1_trn": f1_score(y[i_trn], y_t_trn),
                "lima_trn": lima_score(y[i_trn], y_t_trn, p_minus),
                "accuracy_tst": accuracy_score(y[i_tst], y_t_tst), # clean test performances
                "f1_tst": f1_score(y[i_tst], y_t_tst),
                "lima_tst": lima_score(y[i_tst], y_t_tst, p_minus),
            })
        df = pd.DataFrame(results)
        df['p_minus'] = p_minus
        df['p_plus'] = p_plus
        df['dataset'] = dataset
        df['trial'] = trial
        df.to_csv(output_path)
        print(f"{df.shape[0]} results succesfully stored at {output_path}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('p_minus', type=float, help='noise rate of the negative class')
    parser.add_argument('p_plus', type=float, help='noise rate of the positive class')
    parser.add_argument('dataset', type=str, help='dataset')
    parser.add_argument('--trial', type=int, default=0, metavar='N',
                        help='the trial to plot (default: 0)')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument('--n_folds', type=int, default=10, metavar='N',
                        help='number of cross validation folds (default: 10)')
    parser.add_argument('--n_repetitions', type=int, default=10, metavar='N',
                        help='number of repetitions of the cross validation (default: 10)')
    args = parser.parse_args()
    main(
        args.output_path,
        args.p_minus,
        args.p_plus,
        args.dataset,
        args.trial,
        args.seed,
        args.n_folds,
        args.n_repetitions,
    )
