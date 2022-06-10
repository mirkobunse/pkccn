# pkccn

This package implements various algorithms for handling class-conditional label noise (CCN), also in cases where the class-wise noise rates are partially known.

```
pip install .
```

This package is part of our supplementary material for "Imbalanced Classification with Partially-Known Class-Conditional Label Noise", our submission to ICDM 2022.

## Additional experimental results

The supplementary results of our submission are given as PDF files in the `results/` directory.

- <a href="https://anonymous.4open.science/api/repo/pkccn-5C53/file/results/tables.pdf" target="_blank">results/tables.pdf</a>: F1 scores and f_alpha scores for each individual combination of method, noise configuration, and data set.

## Usage

To make any soft classifier from [scikit-learn](https://scikit-learn.org/stable/) CCN-aware, you only need to wrap it in a `ThresholdedClassifier`.

```python
from pkccn import ThresholdedClassifier

ccn_classifier = ThresholdedClassifier(
    sklearn_base_classifier, # e.g. LogisticRegression()
    method, # "lima", "menon", "mithal", or "default"
    method_args = {"p_minus": 0.5, "verbose": True} # optional arguments
)

ccn_classifier.fit(X_train, y_train)
y_pred = ccn_classifier.predict(X_test)
```

We recommend to use bagging classifiers, like random forests. These classifiers are able to optimize their CCN-aware decision threshold consistently, on out-of-bag noisy labels.

To make use of this feature, you need to specify `prediction_method="oob"` and ask the classifier to compute out-of-bag scores.

```python
ccn_classifier = ThresholdedClassifier(
    RandomForestClassifier(oob_score=True),
    method,
    prediction_method = "oob"
)
```

## Experiments

Experiments are handled with GNU Make. To run them locally, just call

```
make
```

in your terminal. To inspect the process without running it, call `make -n` (dry run). Our experiments have the following requirements:

- Python 3.10
- Julia 1.6 (optional, for plotting the CD diagrams)
- LuaLaTeX (optional, for plotting the CD diagrams)
- Docker (optional, if you prefer to install the above dependencies in an isolated container)

To build a Docker image with all of the dependencies installed, call `make` from the `docker/` directory and start a container with the `docker/run.sh` script.

```
cd docker/
make
./run.sh
```

## Development / unit testing

During development, run tests locally with the `unittest` package.

```
python -m venv venv
venv/bin/pip install .[tests]
venv/bin/python -m unittest
```
