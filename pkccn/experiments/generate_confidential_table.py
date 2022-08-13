import argparse
import numpy as np
import pandas as pd

def main(
        output_path,
        fact_path = "results/fact.csv",
        crab_path = "results/confidential_crab.csv",
    ):
    df_fact = pd.read_csv(fact_path, sep='\s*,\s*', engine='python') # i f*cking hate pandas
    df_crab = pd.read_csv(crab_path, sep='\s*,\s*', engine='python')

    # generate a LaTeX table
    print(f"Generating {output_path}")
    with open(output_path, "w") as f:
        print("\\begin{tabular}{lccc}", file=f)
        print("  \\toprule", file=f)
        print("  \\makecell[lc]{method} & \\makecell{CCN labels\\\\(open)} & \\makecell{CCN labels\\\\(closed)} & \\makecell{clean labels\\\\(SOTA)} \\\\", file=f)
        print("  \\midrule", file=f)
        for m in [
                "Li \& Ma tree (ours; PK-CCN)",
                "Li \& Ma threshold (ours; PK-CCN)",
                "Menon et al. (2015; PK-CCN; F1 score)",
                "Menon et al. (2015; CU-CCN; F1 score)",
                "Mithal et al. (2017; CU-CCN; G measure)",
                "default (F1 score)"
                ]:
            df_open = df_fact[(df_fact["method"]==m) & ( # select CCN open data outcome
                (df_fact["classifier"]=="LiMaForestClassifier") |
                (df_crab["classifier"]=="LiMaRandomForest") |
                (df_fact["classifier"]=="RandomForestClassifier")
            )]
            if len(df_open) != 1: # make sure everything works alright
                print(f"df_open (len {len(df_open)}); m = {m}")
            df_closed = df_crab[(df_crab["method"]==m) & ( # select CCN closed data outcome
                (df_crab["classifier"]=="LiMaForestClassifier") |
                (df_crab["classifier"]=="LiMaRandomForest") |
                (df_crab["classifier"]=="RandomForestClassifier")
            )]
            if len(df_closed) != 1:
                print(f"df_closed (len {len(df_closed)}); m = {m}")
            df_clean = df_crab[(df_crab["method"]==m) & (df_crab["classifier"]=="SotaClassifier")]
            ccn_open = df_open['lima'].values[0]
            ccn_closed = df_closed['lima'].values[0]
            clean = df_clean['lima'].values[0] if len(df_clean) == 1 else -np.inf # The LiMaRandomForest cannot be evaluated on clean labels
            candidates = np.round([ccn_open, ccn_closed, clean], decimals=2)
            best = np.flatnonzero(candidates == candidates.max()) # all best values
            ccn_open = f"${ccn_open:.2f} \\pm {df_open['lima_std'].values[0]:.2f}$"
            ccn_closed = f"${ccn_closed:.2f} \\pm {df_closed['lima_std'].values[0]:.2f}$"
            if len(df_clean) != 1:
                clean = "--" # The LiMaRandomForest cannot be evaluated on clean labels
            else:
                clean = f"${clean:.2f} \\pm {df_clean['lima_std'].values[0]:.2f}$"
                df_clean2 = df_fact[np.logical_and( # check equivalence of both experiments
                    df_fact["method"] == m,
                    df_fact["classifier"] == "SotaClassifier"
                )]
                if clean != (f"${df_clean2['lima'].values[0]:.2f} \\pm {df_clean2['lima_std'].values[0]:.2f}$"):
                    print("OPEN: " + f"${df_clean2['lima'].values[0]:.2f} \\pm {df_clean2['lima_std'].values[0]:.2f}$")
                    print("CLOSED: " + clean)
            if 0 in best:
                ccn_open = f"$\\mathbf{{{ccn_open[1:-1]}}}$"
            if 1 in best:
                ccn_closed = f"$\\mathbf{{{ccn_closed[1:-1]}}}$"
            if 2 in best:
                clean = f"$\\mathbf{{{clean[1:-1]}}}$"
            print(m, ccn_open, ccn_closed, clean, sep=" & ", end=" \\\\\n", file=f)
        print("  \\bottomrule", file=f)
        print("\\end{tabular}", file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('fact_path', type=str, help='path of an input *.csv file')
    parser.add_argument('crab_path', type=str, help='path of an input *.csv file')
    args = parser.parse_args()
    main(args.output_path, args.fact_path, args.crab_path)
