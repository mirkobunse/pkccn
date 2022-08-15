import argparse
import numpy as np
import pandas as pd

def main(output_path, mrk_path):
    df_mrk = pd.read_csv(mrk_path, sep='\s*,\s*', engine='python')

    # generate a LaTeX table
    print(f"Generating {output_path}")
    with open(output_path, "w") as f:
        print("\\begin{tabular}{lccc}", file=f)
        print("  \\toprule", file=f)
        print("  \\makecell[lc]{method} & \\makecell{CCN labels\\\\(closed)} & \\makecell{clean labels\\\\(SOTA)} \\\\", file=f)
        print("  \\midrule", file=f)
        for m in [
                "Li \& Ma tree (ours; PK-CCN)",
                "Li \& Ma threshold (ours; PK-CCN)",
                "Menon et al. (2015; PK-CCN; F1 score)",
                "Menon et al. (2015; CU-CCN; F1 score)",
                "Mithal et al. (2017; CU-CCN; G measure)",
                "default (F1 score)"
                ]:
            df_closed = df_mrk[(df_mrk["method"]==m) & ( # select CCN closed data outcome
                (df_mrk["classifier"]=="LiMaRandomForest") |
                (df_mrk["classifier"]=="RandomForestClassifier")
            )]
            if len(df_closed) != 1:
                print(f"df_closed (len {len(df_closed)}); m = {m}")
            df_clean = df_mrk[(df_mrk["method"]==m) & (df_mrk["classifier"]=="SotaClassifier")]
            ccn_closed = df_closed['lima'].values[0]
            clean = df_clean['lima'].values[0] if len(df_clean) == 1 else -np.inf # The LiMaRandomForest cannot be evaluated on clean labels
            candidates = np.round([ccn_closed, clean], decimals=2)
            best = np.flatnonzero(candidates == candidates.max()) # all best values
            ccn_closed = f"${ccn_closed:.2f} \\pm {df_closed['lima_std'].values[0]:.2f}$"
            if len(df_clean) != 1:
                clean = "--" # The LiMaRandomForest cannot be evaluated on clean labels
            else:
                clean = f"${clean:.2f} \\pm {df_clean['lima_std'].values[0]:.2f}$"
            if 0 in best:
                ccn_closed = f"$\\mathbf{{{ccn_closed[1:-1]}}}$"
            if 1 in best:
                clean = f"$\\mathbf{{{clean[1:-1]}}}$"
            print(m, ccn_closed, clean, sep=" & ", end=" \\\\\n", file=f)
        print("  \\bottomrule", file=f)
        print("\\end{tabular}", file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('mrk_path', type=str, help='path of an input *.csv file')
    args = parser.parse_args()
    main(args.output_path, args.mrk_path)
