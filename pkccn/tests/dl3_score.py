import numpy as np
import pandas as pd
from fact.analysis.statistics import li_ma_significance
from fact.io import read_data as fact_read_data
from pkccn import lima_score, lima_threshold

# read DL3 data with gamma predictions
dl3 = fact_read_data("data/crab_dl3.hdf5", "events") # obtain crab_dl3 from https://github.com/fact-project/open_crab_sample_analysis
y_dl3 = dl3["gamma_prediction"].values

# read "on" and "off" annotations
theta_cut = np.sqrt(0.025)
is_on = (dl3.theta_deg < theta_cut).values
is_off = pd.concat([dl3[f'theta_deg_off_{i}'] < theta_cut for i in range(1, 6)], axis=1).values
is_annotated = np.logical_or(is_on, np.any(is_off, axis=1))
y_dl3 = y_dl3[is_annotated] # only consider annotated instances
is_on = is_on[is_annotated]

# lima_score (ours) vs li_ma_significance (gamma ray astronomy)
for threshold in [ 0.8, 0.85, lima_threshold(is_on, y_dl3, 1/6) ]:
    N_on = np.sum(is_on[y_dl3 > threshold])
    N_off = np.sum(y_dl3 > threshold) - N_on
    theirs = li_ma_significance(N_on, N_off, 1/5) # alpha=1/5 <=> p_minus=1/6
    ours = lima_score(is_on[y_dl3 > threshold], np.ones(np.sum(y_dl3 > threshold)), 1/6)
    print(f"t={threshold:.4f}: theirs={theirs:.1f}, ours={ours:.1f}")

#
# Output:
#
# t=0.8000: theirs=26.5, ours=26.5
# t=0.8500: theirs=24.8, ours=24.8
# t=0.8023: theirs=26.6, ours=26.6
#
