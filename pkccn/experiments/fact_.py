import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pkccn import lima_score, Threshold
from fact.io import read_data as fact_read_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from tqdm.auto import tqdm

DATA_DIR = "data/"
HEADER = ["concentration_cog", "concentration_core", "concentration_one_pixel", "concentration_two_pixel", "leakage1", "leakage2", "size", "width", "length", "skewness_long", "skewness_trans", "kurtosis_long", "kurtosis_trans", "num_islands", "num_pixel_in_shower", "photoncharge_shower_variance", "area", "log_size", "size_area", "area_size_cut_var"]



def read_data():
    """Load real-world data. Joins DL2 and DL3 files and computes auxiliary features"""
    dl3 = [x for x in os.listdir(DATA_DIR) if x.endswith("_dl3.hdf5")].pop()
    dl3 = fact_read_data(os.path.join(DATA_DIR, dl3), "events")
    dl2 = [x for x in os.listdir(DATA_DIR) if x.endswith("_dl2.hdf5")].pop()
    dl2 = fact_read_data(os.path.join(DATA_DIR, dl2), "events")
    dl3 = dl3.rename({"event_num": "event"}, axis='columns')
    dl2 = dl2.rename({"event_num": "event"}, axis='columns')
    dl3 = dl3.merge(dl2, on=["event", "night", "run_id"], how="inner")
    dl3["area"] = dl3["width"] * dl3["length"] * np.pi
    dl3["log_size"] = np.log(dl3["size"])
    dl3["size_area"] = dl3["size"] / (dl3["width"] * dl3["length"] * np.pi)
    dl3["area_size_cut_var"] = (dl3["width"] * dl3["length"] * np.pi) / (np.log(dl3["size"]) ** 2)
    transformer = QuantileTransformer(output_distribution="uniform").fit(dl3[HEADER])
    dl3[HEADER] = transformer.transform(dl3[HEADER])
    return dl3

def extract_weak_labels(data, theta2_cut=0.025):
    """Extract noisy On/Off region labels from a DataFrame."""
    X = data[HEADER].values
    theta_cut = np.sqrt(theta2_cut)
    is_on = (data.theta_deg < theta_cut).values
    is_off = pd.concat([data[f'theta_deg_off_{i}'] < theta_cut for i in range(1, 6)], axis=1).values

    sample = np.logical_or(is_on, np.any(is_off, axis=1)) # subsample
    X = X[sample]
    y = is_on[sample] * 2 - 1
    return X, y, sample

def trial(trial_seed, n_folds, p_minus, p_plus, methods, clf, data):
    """A single trial of imblearn.main()"""
    np.random.seed(trial_seed)

    X, y, sample = extract_weak_labels(data)
    data = data.icol[sample]

    # cross_val_predict, fitting a separate threshold in each fold
    y_pred = { m: np.zeros_like(y) for m in methods.keys() } # method name -> predictions
    thresholds = { m: [] for m in methods.keys() } # method name -> thresholds
    day = pd.to_datetime(data['timestamp_y'], unit="s").dt.dayofyear
    group = LabelEncoder().fit_transform(day.values.reshape(-1, 1))
    for i_trn, i_tst in GroupKFold(len(np.unique(day.values))).split(X, y, group):
        clf.fit(X[i_trn,:], y[i_trn])
        y_trn = clf.oob_decision_function_[:,1]
        y_tst = clf.predict_proba(X[i_tst,:])[:,1]
        for method_name, method in methods.items():
            threshold = method(y[i_trn], y_trn)
            y_pred[method_name][i_tst] = (y_tst > threshold).astype(int) * 2 - 1 # in [-1, 1]
            thresholds[method_name].append(threshold)

    # evaluate all predictions
    trial_results = []
    for method_name, y_method in y_pred.items():
        trial_results.append({
            "dataset": "fact",
            "method": method_name,
            "threshold": np.mean(thresholds[method_name]),
            "trial_seed": trial_seed,
            "accuracy": accuracy_score(y, y_method),
            "f1": f1_score(y, y_method),
            "lima": lima_score(y, y_method, p_minus), # noisy LiMa
        })
    return trial_results