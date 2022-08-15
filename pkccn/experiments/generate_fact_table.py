import argparse
import numpy as np
import pandas as pd

def main(input_path, output_path, bold=True, cv=False, thesis=False):
    df = pd.read_csv(input_path, sep='\s*,\s*', engine='python')

    # generate a LaTeX table
    print(f"Generating {output_path}")
    with open(output_path, "w") as f:
        print("\\begin{tabular}{lccc}", file=f)
        print("  \\toprule", file=f)
        if cv:
            print("  \\makecell[lc]{method} & \\makecell{CCN labels\\\\(open; CV)} & \\makecell{clean labels\\\\(SOTA; CV)} \\\\", file=f)
        else:
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
            df_ccn = df[(df["method"]==m) & ( # select CCN closed data outcome
                (df["classifier"]=="LiMaRandomForest") |
                (df["classifier"]=="RandomForestClassifier")
            )]
            if len(df_ccn) != 1:
                print(f"df_ccn (len {len(df_ccn)}); m = {m}")
            df_cln = df[(df["method"]==m) & (df["classifier"]=="SotaClassifier")]
            ccn = df_ccn['lima'].values[0]
            cln = df_cln['lima'].values[0] if len(df_cln) == 1 else -np.inf # The LiMaRandomForest cannot be evaluated on clean labels
            candidates = np.round([ccn, cln], decimals=2)
            best = np.flatnonzero(candidates == candidates.max()) # all best values
            ccn = f"${ccn:.2f} \\pm {df_ccn['lima_std'].values[0]:.2f}$"
            if len(df_cln) != 1:
                cln = "--" # The LiMaRandomForest cannot be evaluated on clean labels
            else:
                cln = f"${cln:.2f} \\pm {df_cln['lima_std'].values[0]:.2f}$"
            if bold and 0 in best:
                ccn = f"$\\mathbf{{{ccn[1:-1]}}}$"
            if bold and 1 in best:
                cln = f"$\\mathbf{{{cln[1:-1]}}}$"
            method = m
            if thesis:
                method = {
                    "Li \& Ma tree (ours; PK-CCN)": "Li \& Ma tree (ours; Alg.~\\ref{alg:ccn:threshold}; PK-CCN)",
                    "Li \& Ma threshold (ours; PK-CCN)": "Li \& Ma threshold (ours; Alg.~\\ref{alg:ccn:tree}; PK-CCN)",
                    "Menon et al. (2015; PK-CCN; F1 score)": "Menon et al. \\cite{menon2015learning} (Alg.~\\ref{alg:ccn:f1}; PK-CCN; $F_1$ score)",
                    "Menon et al. (2015; CU-CCN; F1 score)": "Menon et al. \\cite{menon2015learning} (Alg.~\\ref{alg:ccn:f1}; CU-CCN; $F_1$ score)",
                    "Mithal et al. (2017; CU-CCN; G measure)": "Mithal et al. \\cite{mithal2017rapt} (CU-CCN; G measure)",
                    "default (F1 score)": "$F_1$ threshold unaware of CCN \\cite{koyejo2014consistent,narasimhan2014statistical}",
                }[m]
            print(method, ccn, cln, sep=" & ", end=" \\\\\n", file=f)
        print("  \\bottomrule", file=f)
        print("\\end{tabular}", file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='path of an input *.csv file')
    parser.add_argument('output_path', type=str, help='path of an output *.tex file')
    parser.add_argument("--no_bold", action="store_true", help="whether to omit bold printing")
    parser.add_argument("--cv", action="store_true", help="whether to display CV results")
    parser.add_argument("--thesis", action="store_true", help="whether to display PhD thesis names")
    args = parser.parse_args()
    main(args.input_path, args.output_path, not args.no_bold, args.cv, args.thesis)
