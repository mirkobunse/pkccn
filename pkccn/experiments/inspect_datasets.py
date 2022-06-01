import argparse
import numpy as np
import os
import pandas as pd
from imblearn.datasets import fetch_datasets
from pkccn.experiments import MLPClassifier
from sklearn.model_selection import GridSearchCV

def main(
        output_path,
        seed = 867,
        n_folds = 10
    ):
    print(f"Starting to inspect_datasets, producing {output_path} with seed {seed}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ensure that the directory exists
    np.random.seed(seed)

    # set up the hyper-parameter grid of the base classifier
    cv = GridSearchCV(
        MLPClassifier(class_weight="balanced", hidden_layer_sizes=(50,)),
        { "alpha": [1e-0, 1e-1, 1e-2, 1e-3] },
        scoring = "roc_auc",
        cv = n_folds,
        verbose = 3,
        refit = False
    )

    # evaluate the base classifier on all imblearn data sets
    results = []
    datasets = fetch_datasets().items()
    for i, (dataset, imblearn_dataset) in enumerate(datasets):
        X = imblearn_dataset.data
        y = imblearn_dataset.target
        cv.fit(X, y)
        results.append({
            "dataset": dataset,
            "auroc": cv.best_score_,
            "N": X.shape[0],
            "N_min": np.sum(y == 1),
            "F": X.shape[1],
        })
        print(f"[{i+1}/{len(datasets)}] {dataset} yields AUROC={cv.best_score_:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(output_path)
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument('--n_folds', type=int, default=10, metavar='N',
                        help='number of cross validation folds (default: 10)')
    args = parser.parse_args()
    main(
        args.output_path,
        args.seed,
        args.n_folds,
    )
