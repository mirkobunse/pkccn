import argparse
import numpy as np
import pandas as pd

def main(output_path, fake_path):
    df_fake = pd.read_csv(fake_path, sep='\s*,\s*', engine='python')

    # generate a LaTeX table
    print(f"Generating {output_path}")
    with open(output_path, "w") as f:
        print("\\begin{tabular}{lccc}", file=f)
        print("  \\toprule", file=f)
        print("  \\makecell[lc]{method} & \\makecell{CCN labels\\\\(open; CV)} & \\makecell{clean labels\\\\(SOTA)} \\\\", file=f)
        print("  \\midrule", file=f)
        for m in [
                "Li \& Ma tree (ours; PK-CCN)",
                "Li \& Ma threshold (ours; PK-CCN)",
                "Menon et al. (2015; PK-CCN; F1 score)",
                "Menon et al. (2015; CU-CCN; F1 score)",
                "Mithal et al. (2017; CU-CCN; G measure)",
                "default (F1 score)"
                ]:
            df_ccn = df_fake[(df_fake["method"]==m) & ( # select CCN closed data outcome
                (df_fake["classifier"]=="LiMaRandomForest") |
                (df_fake["classifier"]=="RandomForestClassifier")
            )]
            if len(df_ccn) != 1:
                print(f"df_ccn (len {len(df_ccn)}); m = {m}")
            df_clean = df_fake[(df_fake["method"]==m) & (df_fake["classifier"]=="SotaClassifier")]
            print(m,
                f"${df_ccn['lima'].values[0]:.3f} \\pm {df_ccn['lima_std'].values[0]:.3f}$",
                f"${df_clean['lima'].values[0]:.3f} \\pm {df_clean['lima_std'].values[0]:.3f}$" if len(df_clean) == 1 else "--",
                sep = " & ",
                end = " \\\\\n",
                file = f
            )
        print("  \\bottomrule", file=f)
        print("\\end{tabular}", file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('fake_path', type=str, help='path of an input *.csv file')
    args = parser.parse_args()
    main(args.output_path, args.fake_path)
