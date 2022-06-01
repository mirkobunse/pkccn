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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, GridSearchCV, RepeatedStratifiedKFold
from tqdm.auto import tqdm

def main(
        output_path,
        p_minus,
        p_plus,
        trial = 0,
        seed = 867,
        n_folds = 10,
        n_repetitions = 10,
        is_test_run = False,
    ):
    print(f"Starting to inspect_objectives, producing {output_path} with seed {seed}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # set up the hyper-parameter grid of the base classifier
    cv = GridSearchCV(
        MLPClassifier(class_weight="balanced", hidden_layer_sizes=(50,)),
        { "alpha": [1e-0, 1e-1, 1e-2, 1e-3] },
        scoring = "roc_auc",
        cv = 3,
        verbose = 3,
    )
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repetitions)

    # iterate over all data sets
    results = []
    for i_dataset, dataset in enumerate(datasets(is_test_run)):
        print(f"[{i_dataset+1}/{len(datasets(is_test_run))}] Evaluating {dataset}")
        imblearn_dataset = fetch_datasets()[dataset]
        X = imblearn_dataset.data
        y = imblearn_dataset.target

        i_trn, i_tst = __get_index_of_generator(rskf.split(X, y), trial)
        y_trn = inject_ccn(y[i_trn], p_minus, p_plus)
        y_tst = inject_ccn(y[i_tst], p_minus, p_plus)
        cv.fit(X[i_trn,:], y_trn) # hyper-parameter optimization on noisy data
        print(pd.DataFrame(cv.cv_results_)[["param_alpha", "mean_fit_time", "mean_test_score", "std_test_score"]])
        clf = cv.best_estimator_
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
        dataset_results = []
        for t in np.arange(0., 1.001, step=0.001):
            y_t_trn = (y_pred_trn > t).astype(int) * 2 - 1 # in [-1, 1]
            y_t_tst = (y_pred_tst > t).astype(int) * 2 - 1
            dataset_results.append({
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
        dataset_df = pd.DataFrame(dataset_results)
        dataset_df['p_minus'] = p_minus
        dataset_df['p_plus'] = p_plus
        dataset_df['dataset'] = dataset
        dataset_df['trial'] = trial
        dataset_path = output_path.replace(".csv", f"_{dataset}.csv")
        dataset_df.to_csv(dataset_path)
        print(f"{dataset_df.shape[0]} results succesfully stored at {dataset_path}")
        results.append(dataset_df)
    df = pd.concat(results) # all DataFrames
    df.to_csv(output_path)
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

def __get_index_of_generator(generator, index):
    """Get generator[index] when the generator does not allow indexing."""
    for i, item in enumerate(generator):
        if i != index:
            return item

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('p_minus', type=float, help='noise rate of the negative class')
    parser.add_argument('p_plus', type=float, help='noise rate of the positive class')
    parser.add_argument('--trial', type=int, default=0, metavar='N',
                        help='the trial to plot (default: 0)')
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
        args.trial,
        args.seed,
        args.n_folds,
        args.n_repetitions,
        args.is_test_run,
    )
